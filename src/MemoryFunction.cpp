#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <algorithm>
#include <cstdint>
#include "AlignedAlloc.h"
#include "MemoryFunction.h"

// clang-format off

#define BLOCK_ALIGN 0x10

#ifdef _WIN32
	#define MEMFUNC_USE_WIN32
#elif defined(__APPLE__)
	#include "TargetConditionals.h"
	#include <libkern/OSCacheControl.h>

	#if TARGET_OS_OSX
		#define MEMFUNC_USE_MMAP
		#define MEMFUNC_MMAP_ADDITIONAL_FLAGS (MAP_JIT)
		#if TARGET_CPU_ARM64
			#define MEMFUNC_MMAP_REQUIRES_JIT_WRITE_PROTECT
		#endif
	#elif TARGET_OS_IPHONE && TARGET_CPU_ARM64
		// iOS 26 on TXM/SPTM hardware (A15+): the process itself can never make a
		// page executable - MAP_JIT returns EPERM without dynamic-codesigning, and
		// mprotect silently strips PROT_EXEC. Only a write performed *through an
		// attached debug connection* marks a page as JIT-executable. StikDebug's
		// universal.js implements that debugger side; the app has to ask for it by
		// trapping with brk #0xf00d (x16 = command). See StikJIT INTEGRATION.md.
		#define MEMFUNC_USE_MMAP
		#define MEMFUNC_IOS26_JIT_PROTOCOL
	#else
		#define MEMFUNC_USE_MACHVM
		#if TARGET_OS_IPHONE
			#define MEMFUNC_MACHVM_STRICT_PROTECTION
		#endif
	#endif
#elif defined(__EMSCRIPTEN__)
	#include <emscripten.h>
	#define MEMFUNC_USE_WASM
#else
	#define MEMFUNC_USE_MMAP
#endif

#if defined(MEMFUNC_USE_WIN32)
#include <windows.h>
#elif defined(MEMFUNC_USE_MACHVM)
#include <mach/mach_init.h>
#include <mach/vm_map.h>
#elif defined(MEMFUNC_USE_MMAP)
#include <sys/mman.h>
#include <pthread.h>
#if defined(MEMFUNC_IOS26_JIT_PROTOCOL)
#include <unistd.h>

// --- iOS 26 TXM JIT protocol (StikDebug / StikJIT "universal" script) ---------
// The attached debugger blesses a region by writing one byte into each 16K page
// over the debug connection; that is what makes the pages executable on TXM
// hardware. The app requests it by trapping with brk #0xf00d, passing the
// command in x16 and the region in x0/x1. A brk with no debugger attached kills
// the process, so every call is gated on CS_DEBUGGED.
extern "C" int csops(pid_t pid, unsigned int ops, void* useraddr, size_t usersize);

static bool MemFunc_IsDebuggerAttached()
{
	uint32_t flags = 0;
	if(csops(getpid(), 0 /*CS_OPS_STATUS*/, &flags, sizeof(flags)) != 0) return false;
	return (flags & 0x10000000u) != 0; //CS_DEBUGGED
}

__attribute__((noinline, optnone, naked))
static void* JIT26PrepareRegion(void* address, size_t length)
{
	__asm__ volatile(
	    "mov x16, #1\n"
	    "brk #0xf00d\n"
	    "ret\n");
}

// A region can only be blessed while the JIT script is still attached, and the
// script's window closes early in the launch sequence ("a region introduced
// later cannot be prepared by a script that has already detached"). Play!
// creates a new executable region for every basic block, so instead of blessing
// per block we reserve one large arena up front, bless it once, and sub-allocate
// every block out of it.
#include <mutex>
#include <vector>
#include <cstdio>
#include <errno.h>

static const size_t MEMFUNC_JIT_ARENA_SIZE = 96 * 1024 * 1024;

namespace
{
	struct MEMFUNC_FREE_CHUNK
	{
		uint8* ptr;
		size_t size;
	};
}

static uint8* g_jitArenaBase = nullptr;
static size_t g_jitArenaBump = 0;
static bool g_jitArenaBlessed = false;
static std::mutex g_jitArenaMutex;
static std::vector<MEMFUNC_FREE_CHUNK> g_jitArenaFree;
static char g_jitStatus[224] = "jit: arena not initialized";

static void MemFunc_ArenaInitLocked()
{
	if(g_jitArenaBase != nullptr) return;
	void* mem = mmap(nullptr, MEMFUNC_JIT_ARENA_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC,
	                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if(mem == MAP_FAILED)
	{
		snprintf(g_jitStatus, sizeof(g_jitStatus), "jit: arena mmap FAILED errno=%d", errno);
		return;
	}
	g_jitArenaBase = static_cast<uint8*>(mem);
	bool debugged = MemFunc_IsDebuggerAttached();
	if(debugged)
	{
		JIT26PrepareRegion(g_jitArenaBase, MEMFUNC_JIT_ARENA_SIZE);
		g_jitArenaBlessed = true;
	}
	snprintf(g_jitStatus, sizeof(g_jitStatus), "jit: arena %zuMB @%p dbg=%d blessed=%d",
	         static_cast<size_t>(MEMFUNC_JIT_ARENA_SIZE >> 20), static_cast<void*>(g_jitArenaBase),
	         debugged ? 1 : 0, g_jitArenaBlessed ? 1 : 0);
}

static void* MemFunc_ArenaAlloc(size_t size)
{
	std::lock_guard<std::mutex> lock(g_jitArenaMutex);
	MemFunc_ArenaInitLocked();
	if(!g_jitArenaBlessed) return nullptr;
	size_t aligned = (size + (BLOCK_ALIGN - 1)) & ~static_cast<size_t>(BLOCK_ALIGN - 1);
	for(size_t i = 0; i < g_jitArenaFree.size(); i++)
	{
		if(g_jitArenaFree[i].size >= aligned)
		{
			uint8* result = g_jitArenaFree[i].ptr;
			if(g_jitArenaFree[i].size >= aligned + BLOCK_ALIGN)
			{
				g_jitArenaFree[i].ptr += aligned;
				g_jitArenaFree[i].size -= aligned;
			}
			else
			{
				g_jitArenaFree.erase(g_jitArenaFree.begin() + i);
			}
			return result;
		}
	}
	if((g_jitArenaBump + aligned) > MEMFUNC_JIT_ARENA_SIZE) return nullptr;
	uint8* result = g_jitArenaBase + g_jitArenaBump;
	g_jitArenaBump += aligned;
	return result;
}

static bool MemFunc_ArenaOwns(void* ptr)
{
	return (g_jitArenaBase != nullptr) && (ptr >= g_jitArenaBase) &&
	       (ptr < (g_jitArenaBase + MEMFUNC_JIT_ARENA_SIZE));
}

static void MemFunc_ArenaFree(void* ptr, size_t size)
{
	std::lock_guard<std::mutex> lock(g_jitArenaMutex);
	size_t aligned = (size + (BLOCK_ALIGN - 1)) & ~static_cast<size_t>(BLOCK_ALIGN - 1);
	g_jitArenaFree.push_back({static_cast<uint8*>(ptr), aligned});
}

// Called by the app as early as possible (while the JIT script is still
// attached) so the arena gets blessed before any game code is compiled.
extern "C" void MemFunc_InitJitArena()
{
	std::lock_guard<std::mutex> lock(g_jitArenaMutex);
	MemFunc_ArenaInitLocked();
}

extern "C" const char* MemFunc_GetJitStatus()
{
	return g_jitStatus;
}
#endif
#elif defined(MEMFUNC_USE_WASM)
EM_JS_DEPS(WasmMemoryFunction, "$addFunction,$removeFunction");
EM_JS(int, WasmCreateFunction, (emscripten::EM_VAL moduleHandle),
{
	let module = Emval.toValue(moduleHandle);
	let moduleInstance = new WebAssembly.Instance(module, {
		env: {
			memory: wasmMemory,
			fctTable : Module.codeGenImportTable
		}
	});
	let fct = moduleInstance.exports.codeGenFunc;
	let fctId = addFunction(fct, 'vi');
	return fctId;
});
EM_JS(void, WasmDeleteFunction, (int fctId),
{
	removeFunction(fctId);
});
EM_JS(emscripten::EM_VAL, WasmCreateModule, (uintptr_t code, uintptr_t size),
{
	//var fs = require('fs');
	let moduleBytes = HEAP8.subarray(code, code + size);
	//fs.writeFileSync('module.wasm', moduleBytes);
	//{
	//	let bytesCopy = new Uint8Array(moduleBytes);
	//	let blob = new Blob([bytesCopy], { type: "binary/octet-stream" });
	//	let url = URL.createObjectURL(blob);
	//	console.log(url);
	//}
	let module = new WebAssembly.Module(moduleBytes);
	return Emval.toHandle(module);
});
#else
#error "No API to use for CMemoryFunction"
#endif

CMemoryFunction::CMemoryFunction()
: m_code(nullptr)
, m_size(0)
{

}

CMemoryFunction::CMemoryFunction(const void* code, size_t size)
: m_code(nullptr)
{
#if defined(MEMFUNC_USE_WIN32)
	m_size = size;
	m_code = framework_aligned_alloc(size, BLOCK_ALIGN);
	memcpy(m_code, code, size);
	
	DWORD oldProtect = 0;
	BOOL result = VirtualProtect(m_code, size, PAGE_EXECUTE_READWRITE, &oldProtect);
	assert(result == TRUE);
#elif defined(MEMFUNC_USE_MACHVM)
	vm_size_t page_size = 0;
	host_page_size(mach_task_self(), &page_size);
	unsigned int allocSize = ((size + page_size - 1) / page_size) * page_size;
	vm_allocate(mach_task_self(), reinterpret_cast<vm_address_t*>(&m_code), allocSize, TRUE); 
	memcpy(m_code, code, size);
	vm_prot_t protection =
	#ifdef MEMFUNC_MACHVM_STRICT_PROTECTION
		VM_PROT_READ | VM_PROT_EXECUTE;
	#else
		VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE;
	#endif
	kern_return_t result = vm_protect(mach_task_self(), reinterpret_cast<vm_address_t>(m_code), size, 0, protection);
	assert(result == 0);
	m_size = allocSize;
#elif defined(MEMFUNC_USE_MMAP)
	uint32 additionalMapFlags = 0;
	#ifdef MEMFUNC_MMAP_ADDITIONAL_FLAGS
		additionalMapFlags = MEMFUNC_MMAP_ADDITIONAL_FLAGS;
	#endif
	m_size = size;
#ifdef MEMFUNC_IOS26_JIT_PROTOCOL
	// Blocks come out of the pre-blessed arena. Only if that is unavailable do we
	// fall back to a private mapping (which can only run if the script happens to
	// still be attached).
	m_code = MemFunc_ArenaAlloc(size);
	if(m_code == nullptr)
	{
		m_code = mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		assert(m_code != MAP_FAILED);
		if(MemFunc_IsDebuggerAttached())
		{
			JIT26PrepareRegion(m_code, m_size);
		}
	}
#else
	m_code = mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS | additionalMapFlags, -1, 0);
	assert(m_code != MAP_FAILED);
#endif
#ifdef MEMFUNC_MMAP_REQUIRES_JIT_WRITE_PROTECT
	pthread_jit_write_protect_np(false);
#endif
	memcpy(m_code, code, size);
#ifdef MEMFUNC_MMAP_REQUIRES_JIT_WRITE_PROTECT
	pthread_jit_write_protect_np(true);
#endif
#elif defined(MEMFUNC_USE_WASM)
	m_wasmModule = emscripten::val::take_ownership(WasmCreateModule(reinterpret_cast<uintptr_t>(code), size));
	m_size = size;
	m_code = reinterpret_cast<void*>(WasmCreateFunction(m_wasmModule.as_handle()));
#endif
	ClearCache();
#if !defined(MEMFUNC_USE_WASM)
	assert((reinterpret_cast<uintptr_t>(m_code) & (BLOCK_ALIGN - 1)) == 0);
#endif
}

CMemoryFunction::~CMemoryFunction()
{
	Reset();
}

void CMemoryFunction::ClearCache()
{
#ifdef __APPLE__
	sys_icache_invalidate(m_code, m_size);
#elif defined(MEMFUNC_USE_MMAP)
	#if defined(__arm__) || defined(__aarch64__)
		__clear_cache(m_code, reinterpret_cast<uint8*>(m_code) + m_size);
	#endif
#endif
}

void CMemoryFunction::Reset()
{
	if(m_code != nullptr)
	{
#if defined(MEMFUNC_USE_WIN32)
		framework_aligned_free(m_code);
#elif defined(MEMFUNC_USE_MACHVM)
		vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(m_code), m_size);
#elif defined(MEMFUNC_USE_MMAP)
	#ifdef MEMFUNC_IOS26_JIT_PROTOCOL
		// Arena memory must never be unmapped - it can't be blessed again.
		if(MemFunc_ArenaOwns(m_code))
		{
			MemFunc_ArenaFree(m_code, m_size);
		}
		else
		{
			munmap(m_code, m_size);
		}
	#else
		munmap(m_code, m_size);
	#endif
#elif defined(MEMFUNC_USE_WASM)
		WasmDeleteFunction(reinterpret_cast<int>(m_code));
#endif
	}
	m_code = nullptr;
	m_size = 0;
#if defined(MEMFUNC_USE_WASM)
	m_wasmModule = emscripten::val();
#endif
}

bool CMemoryFunction::IsEmpty() const
{
	return m_code == nullptr;
}

CMemoryFunction& CMemoryFunction::operator =(CMemoryFunction&& rhs)
{
	Reset();
	std::swap(m_code, rhs.m_code);
	std::swap(m_size, rhs.m_size);
#if defined(MEMFUNC_USE_WASM)
	std::swap(m_wasmModule, rhs.m_wasmModule);
#endif
	return (*this);
}

void CMemoryFunction::operator()(void* context)
{
	typedef void (*FctType)(void*);
	auto fct = reinterpret_cast<FctType>(m_code);
	fct(context);
}

void* CMemoryFunction::GetCode() const
{
	return m_code;
}

size_t CMemoryFunction::GetSize() const
{
	return m_size;
}

void CMemoryFunction::BeginModify()
{
#if defined(MEMFUNC_USE_MACHVM) && defined(MEMFUNC_MACHVM_STRICT_PROTECTION)
	kern_return_t result = vm_protect(mach_task_self(), reinterpret_cast<vm_address_t>(m_code), m_size, 0, VM_PROT_READ | VM_PROT_WRITE);
	assert(result == 0);
#elif defined(MEMFUNC_USE_MMAP) && defined(MEMFUNC_MMAP_REQUIRES_JIT_WRITE_PROTECT)
	pthread_jit_write_protect_np(false);
#endif
}

void CMemoryFunction::EndModify()
{
#if defined(MEMFUNC_USE_MACHVM) && defined(MEMFUNC_MACHVM_STRICT_PROTECTION)
	kern_return_t result = vm_protect(mach_task_self(), reinterpret_cast<vm_address_t>(m_code), m_size, 0, VM_PROT_READ | VM_PROT_EXECUTE);
	assert(result == 0);
#elif defined(MEMFUNC_USE_MMAP) && defined(MEMFUNC_MMAP_REQUIRES_JIT_WRITE_PROTECT)
	pthread_jit_write_protect_np(true);
#endif
	ClearCache();
}

CMemoryFunction CMemoryFunction::CreateInstance()
{
#if defined(MEMFUNC_USE_WASM)
	CMemoryFunction result;
	result.m_wasmModule = m_wasmModule;
	result.m_size = m_size;
	result.m_code = reinterpret_cast<void*>(WasmCreateFunction(m_wasmModule.as_handle()));
	return result;
#else
	return CMemoryFunction(GetCode(), GetSize());
#endif
}

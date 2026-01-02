- [ ] Asahi Linux Uart m1m1 or ?? debugging https://fahrplan.events.ccc.de/congress/2025/fahrplan/event/asahi-linux-porting-linux-to-apple-silicon


The Asahi Linux team developed m1n1, a lightweight hypervisor that traces Memory-Mapped I/O (MMIO) accesses to reverse engineer Apple Silicon hardware without disassembling proprietary drivers. This approach allows them to observe how macOS interacts with hardware registers in real-time and understand the protocols needed to write Linux drivers.
​
​

How m1n1 Hypervisor Works
The m1n1 hypervisor runs at Exception Level 2 (EL2) on the ARM64 processor, sitting between the hardware and the operating system (which runs at EL1). It boots macOS or Linux as a guest OS while transparently intercepting hardware accesses. The key technique involves manipulating the virtual memory page tables so that RAM is mapped directly for normal access, but MMIO regions (hardware registers) are deliberately not mapped. When the guest OS tries to access an unmapped MMIO address, the CPU triggers a data abort exception that traps into the hypervisor.
​
​

The hypervisor then logs the access details including the CPU program counter location, whether it's a read or write operation, the data size, and the specific hardware register address being accessed. This creates a complete trace of all hardware interactions without needing to reverse engineer driver assembly code.
​

Python-Based Interactive Environment
What makes m1n1 particularly powerful is its Python integration. The hypervisor works with Python code running on a separate host machine connected via USB or network, allowing researchers to "puppeteer" the M1 remotely. Parts of the hypervisor functionality are actually written in Python, enabling rapid iteration - researchers can even update hypervisor code live during guest execution without rebooting.
​

The hv.py Python script handles hypervisor event processing, MMIO trace events, and exception handling. This lets developers write interactive scripts to capture specific register accesses, experiment with different register values, and identify undocumented hardware functionality by observing macOS behavior patterns.
​

Debugging and Development Features
Beyond MMIO tracing, m1n1 includes standard debugging capabilities like execution breakpoints, single-stepping, and backtraces. It supports both GDB and LLDB debugging protocols, allowing developers to debug Linux kernel code and even m1n1 itself running under the hypervisor. The tool traces synchronous exceptions, system register emulation, and page table operations to provide deep visibility into low-level hardware behavior.
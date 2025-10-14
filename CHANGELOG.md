# Version 1.0.0 (2025-06-10)
- Initial release of the OAAX Nvidia GPUs implementation.
- Includes the OAAX runtime and conversion toolchain.
- There are several runtime versions based on the architecture (x86_64 and aarch64) and CUDA version ( 11.x, and 12.x).
- The conversion toolchain is expected to run on x86_64.

# Version 1.1.0 (2025-07-15)
- Added support for Windows

# Version 1.1.1 (2025-09-03)
- Fix a bug where inference stops working after a few iterations for some models

# Version 1.1.2 (2025-10-14)
- Adds support for Ubuntu 25, by removing the executable flag from the libraries ELF.
- Includes `msvcp140_1.dll` in the Windows artifacts to fix runtime errors on some systems.

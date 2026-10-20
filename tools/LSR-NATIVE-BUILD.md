# LSR native build

LSR is one deployable executable per OS/architecture, not per processing device.
On Windows the same `LSR.exe` supports CPU and CUDA execution. Engine launches
the component and RM supplies its resource lease and execution conditions.
CLI/API configuration selects the runtime processing unit; compilation does
not bind the component to the build host's CPU or GPU.

Component-only builds belong in
`JesterWorkspace/jester-engine/jester-components/built/j-lsr/windows-x64/<version>/router.dist/LSR.exe`.
`built/engine` is not the destination for a new LSR-only build. Historical
builds and their evidence keep their original paths.

`lsr_nuitka_build.py` renders the pinned command for the guarded compiler
launcher. Run it from the independent `code/j-lsr` repository. `--workspace`
means the `JesterWorkspace` directory, and defaults to that ancestor in the
target layout. It does not launch an uncontained compiler:

```powershell
python tools/lsr_nuitka_build.py --python <isolated-build-python> --version <version> --report <absolute-report.xml> --check-runtime
```

Install Core, RM Client and LSR from the canonical checkouts into the isolated
build environment. Use CUDA-enabled PyTorch, which also runs on CPU. A guarded
launcher must call `validate_runtime` before compiling; a CPU-only Torch
installation is rejected. This check reads build metadata without initializing
CUDA. Keep the dependency lock and source hashes beside the compilation report.
Models remain external inputs. Other optional backends (Windows ML/NPU) require
their own packaged dependencies and runtime verification; CUDA support alone
does not qualify them. The Windows recipe produces PE; an ELF artifact must be
built and verified with the corresponding Linux toolchain.

The policy keeps Jester modules compiled and the selected third-party families
as frozen bytecode. Hugging Face Hub remains completely included because its
lazy imports are reachable at runtime. Distribution metadata for the semantic
stack is included explicitly.

`torch/include/**` is excluded at build time. These are development headers for
C++/CUDA extension compilation, which the LSR eager-inference path does not use.
The policy does not remove torch DLLs, NumPy/SciPy extensions, certificates or
LSR resource manifests. Runtime extension compilation and `torch.compile`
are outside this eager-inference packaging profile. CUDA DLLs and runtime
libraries are not redundant headers and must remain in the deployable directory.

Compare the deliverable `.dist` directory with the previous `.dist`, separately
from `.build` intermediates and the build environment. Verify executable help,
component metadata, device enumeration and real offline sign/route operations
with a local encoder before reporting the candidate as functional. CPU and GPU
execution must be verified separately against the same executable.

This recipe is the Standalone Build profile. The Fast Build profile and its
Conda/Micromamba loading contract remain separate implementation work; moving
the source tree does not establish that runtime contract.

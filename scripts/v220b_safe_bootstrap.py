"""Sterile, source-only launcher for the v2.20b sparse-cache preflight.

This file is deliberately self contained.  It is the first repository source
executed by a formal process; its Git blob is checked by an external pre-launch
gate and checked again in-process.  It never discovers repository files.
"""

from __future__ import annotations

# Only interpreter-provided/frozen modules are imported before startup and
# sys.path have been validated.  All filesystem-backed stdlib imports happen
# later, while the project/cwd/script directories are known to be absent.
import builtins
import os
import sys


PROJECT_ROOT = "D:/tiktok/techjam-v2-20b-sparse-cache"
RUNTIME_BASE = "D:/tiktok/.v220b_runtime"
PYTHON_EXECUTABLE = "D:/450/conda/envs/tiktok/python.exe"
PYTHON_EXECUTABLE_BYTES = 93_184
PYTHON_EXECUTABLE_SHA256 = (
    "7819c841b9a6457da034e567563de1283dbc0b86482fd83d62b5d982d2a83a63"
)
PYTHON_VERSION = "3.11.16"
SQLITE_VERSION = "3.53.4"
DLL_DIRECTORY = "D:/450/conda/envs/tiktok/DLLs"
BOOTSTRAP_WORKTREE_PATH = (
    PROJECT_ROOT + "/scripts/v220b_safe_bootstrap.py"
)

RUNNER_PATH = PROJECT_ROOT + "/scripts/probe_sparse_multiview_cache_preflight.py"
RUNNER_MODULE = "scripts.probe_sparse_multiview_cache_preflight"
# Replaced exactly once after the final runner blob exists.  Keeping the value
# conspicuous makes an unpatched implementation fail closed.
RUNNER_BLOB = "f5d08d65da7450fe4fc723fd027dd987dedf2959"
WORKER_PATH = PROJECT_ROOT + "/scripts/sparse_multiview_candidate_worker.py"
WORKER_MODULE = "scripts.sparse_multiview_candidate_worker"
WORKER_BLOB = "b44a0c2cbb4c9b4d34aedd6795dbed1ff24a5020"

ATTESTATION_ATTRIBUTE = "_techjam_v220b_bootstrap_attestation"
MAX_CAPTURE_BYTES = 1 << 20

_STDLIB_PATHS = (
    "D:/450/conda/envs/tiktok/python311.zip",
    "D:/450/conda/envs/tiktok/DLLs",
    "D:/450/conda/envs/tiktok/Lib",
    "D:/450/conda/envs/tiktok",
)

# Only these local modules are executable.  Other pinned files in the prereg
# are semantic inputs which the frozen worker hashes, not import candidates.
_LOCAL_SOURCE_ROWS = {
    "evaluator": (
        "evaluator/__init__.py",
        "36ae92f27844cf6288fd433bdb4b21ca4b2a6b07",
        True,
    ),
    "evaluator.local_evaluator": (
        "evaluator/local_evaluator.py",
        "7c808347b31ef3121a9cbc4810ac3eb325f950ba",
        False,
    ),
    "scripts": (
        "scripts/__init__.py",
        "3571b1ee55a368c944442b55d00adc4aa8d0ebfd",
        True,
    ),
    "scripts.c200_candidate_worker": (
        "scripts/c200_candidate_worker.py",
        "b94fddcf5a9b20ddde540f3f43ea9962982cb096",
        False,
    ),
    "starter": (
        "starter/__init__.py",
        "f4cea264f261c96c4ff2166ad00b28a8025a55e3",
        True,
    ),
    "starter.agent": (
        "starter/agent.py",
        "421c6d43c598102b8fefb181b72bab5da4bf1294",
        False,
    ),
    "starter.attributes": (
        "starter/attributes.py",
        "92260323f077c9861aa4edd5242aff772c875760",
        False,
    ),
    "starter.clarification": (
        "starter/clarification.py",
        "b648571dc873cb790455898a2ae3077cc9760c45",
        False,
    ),
    "starter.coverage": (
        "starter/coverage.py",
        "59a6507fef63afa0d9761323f5771a52741c811a",
        False,
    ),
    "starter.p8_negative": (
        "starter/p8_negative.py",
        "719078234dba297ce59f68d8a2b1734ec53c9c63",
        False,
    ),
    "starter.reranker": (
        "starter/reranker.py",
        "f179e7c75d4c0c5189f3a62139de5f51c8ec231a",
        False,
    ),
    "starter.slot_ledger": (
        "starter/slot_ledger.py",
        "72975cff12af59e4044e52911c58294cd74a785a",
        False,
    ),
    "starter.sparse_multiview": (
        "starter/sparse_multiview.py",
        "4adf065b0384ab5d45f7bd4582bf7aaf217348a5",
        False,
    ),
}

_LOCAL_PREFIXES = ("evaluator", "scripts", "starter")
_V219_RESULT = "small_ranker_v2_19_registry_ca_g0_20260831.json"
_V219_CACHE = "small_ranker_v2_19_registry_ca_g0_cache_20260831"

_ORIGINAL_FUNCTIONS: dict[str, object] = {}
_LEXICAL_GUARD_INSTALLED = False
_DLL_DIRECTORY_HANDLE = None
_CURRENT_ATTESTATION: dict[str, object] | None = None


class BootstrapError(RuntimeError):
    """A fixed-code, non-disclosing bootstrap failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _path_key(value: object) -> str:
    """Return a filesystem-free Windows comparison key."""

    try:
        raw = os.fspath(value)
    except TypeError:
        return ""
    if isinstance(raw, bytes):
        try:
            raw = os.fsdecode(raw)
        except UnicodeError:
            return ""
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        return ""
    text = raw.replace("/", "\\")
    lowered = text.casefold()
    for prefix in ("\\\\?\\unc\\", "\\\\?\\", "\\\\.\\", "\\??\\"):
        if lowered.startswith(prefix):
            if prefix == "\\\\?\\unc\\":
                text = "\\\\" + text[len(prefix) :]
            else:
                text = text[len(prefix) :]
            break
    parts: list[str] = []
    for raw_part in text.split("\\"):
        if not raw_part or raw_part == ".":
            continue
        if raw_part == "..":
            if parts and not parts[-1].endswith(":"):
                parts.pop()
            else:
                parts.append(raw_part)
            continue
        # Win32 normally aliases trailing dots/spaces.  Treat those aliases as
        # the protected spelling without asking the filesystem.
        parts.append(raw_part.rstrip(" .").casefold())
    return "\\".join(parts)


def _path_components(value: object) -> tuple[str, ...]:
    key = _path_key(value)
    return tuple(part for part in key.split("\\") if part)


def _is_v219_denied(value: object) -> bool:
    components = _path_components(value)
    for index in range(max(0, len(components) - 2)):
        if components[index : index + 2] != ("experiments", "fast_track"):
            continue
        candidate = components[index + 2]
        if candidate == _V219_RESULT or candidate.startswith(_V219_RESULT + ":"):
            return True
        if candidate == _V219_CACHE or candidate.startswith(_V219_CACHE + ":"):
            return True
    return False


def _is_fast_track_listing(value: object) -> bool:
    components = _path_components(value)
    return len(components) >= 2 and components[-2:] == (
        "experiments",
        "fast_track",
    )


def _guard_path(value: object, *, listing: bool = False) -> None:
    if _is_v219_denied(value) or (listing and _is_fast_track_listing(value)):
        raise PermissionError("V219_NAMESPACE_DENIED")


def _audit_hook(event: str, arguments: tuple[object, ...]) -> None:
    single_path_events = {
        "open",
        "os.chdir",
        "os.chmod",
        "os.mkdir",
        "os.remove",
        "os.rmdir",
        "os.truncate",
        "os.unlink",
    }
    if event in single_path_events and arguments:
        _guard_path(arguments[0])
    elif event in {"os.listdir", "os.scandir"} and arguments:
        _guard_path(arguments[0], listing=True)
    elif event in {"os.link", "os.rename", "os.replace", "os.symlink"}:
        for value in arguments[:2]:
            _guard_path(value)
    elif event == "subprocess.Popen":
        # CPython supplies executable, argv, cwd, env.  Exact frozen children
        # cannot hide a protected path in their invocation surface.
        for value in arguments[:3]:
            if isinstance(value, (list, tuple)):
                for child in value:
                    _guard_path(child)
            else:
                _guard_path(value)


def _wrap_single_path(function: object, *, listing: bool = False):
    def guarded(path, *args, **kwargs):
        _guard_path(path, listing=listing)
        return function(path, *args, **kwargs)

    return guarded


def _wrap_two_paths(function: object):
    def guarded(source, destination, *args, **kwargs):
        _guard_path(source)
        _guard_path(destination)
        return function(source, destination, *args, **kwargs)

    return guarded


def _install_lexical_guard() -> None:
    global _LEXICAL_GUARD_INSTALLED
    if _LEXICAL_GUARD_INSTALLED:
        raise BootstrapError("LEXICAL_GUARD_REINSTALL")
    sys.addaudithook(_audit_hook)
    _ORIGINAL_FUNCTIONS["builtins.open"] = builtins.open
    builtins.open = _wrap_single_path(builtins.open)  # type: ignore[assignment]
    for name in (
        "access",
        "chdir",
        "chmod",
        "lstat",
        "mkdir",
        "open",
        "readlink",
        "remove",
        "rmdir",
        "stat",
        "truncate",
        "unlink",
    ):
        function = getattr(os, name, None)
        if function is not None:
            _ORIGINAL_FUNCTIONS["os." + name] = function
            setattr(os, name, _wrap_single_path(function))
    for name in ("listdir", "scandir"):
        function = getattr(os, name)
        _ORIGINAL_FUNCTIONS["os." + name] = function
        setattr(os, name, _wrap_single_path(function, listing=True))
    for name in ("link", "rename", "replace", "symlink"):
        function = getattr(os, name, None)
        if function is not None:
            _ORIGINAL_FUNCTIONS["os." + name] = function
            setattr(os, name, _wrap_two_paths(function))
    _LEXICAL_GUARD_INSTALLED = True


def _parse_cli(raw_arguments: list[str]) -> tuple[dict[str, str], tuple[str, ...]]:
    try:
        separator = raw_arguments.index("--")
    except ValueError as error:
        raise BootstrapError("CLI_SEPARATOR") from error
    header = raw_arguments[:separator]
    target_arguments = tuple(raw_arguments[separator + 1 :])
    if len(header) != 10:
        raise BootstrapError("CLI_SHAPE")
    allowed = {
        "--mode",
        "--target-path",
        "--target-module",
        "--target-blob",
        "--bootstrap-blob",
    }
    parsed: dict[str, str] = {}
    for index in range(0, len(header), 2):
        option = header[index]
        value = header[index + 1]
        if option not in allowed or option in parsed or not value:
            raise BootstrapError("CLI_OPTION")
        parsed[option] = value
    if set(parsed) != allowed:
        raise BootstrapError("CLI_REQUIRED")
    return parsed, target_arguments


def _match_target(parsed: dict[str, str]) -> dict[str, str]:
    manifest: list[dict[str, str]] = []
    for mode in ("direct", "module"):
        manifest.append(
            {
                "mode": mode,
                "path": RUNNER_PATH,
                "module": RUNNER_MODULE,
                "blob": RUNNER_BLOB,
            }
        )
        manifest.append(
            {
                "mode": mode,
                "path": WORKER_PATH,
                "module": WORKER_MODULE,
                "blob": WORKER_BLOB,
            }
        )
    observed = (
        parsed["--mode"].casefold(),
        parsed["--target-path"].casefold(),
        parsed["--target-module"].casefold(),
        parsed["--target-blob"].casefold(),
    )
    for row in manifest:
        expected = (
            row["mode"].casefold(),
            row["path"].casefold(),
            row["module"].casefold(),
            row["blob"].casefold(),
        )
        if observed == expected:
            return row
    raise BootstrapError("TARGET_MANIFEST")


def _is_hex_blob(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value.casefold())


def _validate_early_runtime(target: dict[str, str], target_arguments: tuple[str, ...]) -> dict[str, str]:
    if not (
        getattr(sys.flags, "safe_path", False)
        and sys.flags.no_site == 1
        and sys.flags.no_user_site == 1
        and sys.flags.dont_write_bytecode == 1
        and sys.version.split()[0] == PYTHON_VERSION
        and _path_key(sys.executable) == _path_key(PYTHON_EXECUTABLE)
    ):
        raise BootstrapError("PYTHON_FLAGS")
    allowed_python = {
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    if target["mode"] == "module":
        python_path = os.environ.get("PYTHONPATH")
        if not python_path:
            raise BootstrapError("MODULE_PYTHONPATH")
        allowed_python["PYTHONPATH"] = python_path
    elif "PYTHONPATH" in os.environ:
        raise BootstrapError("DIRECT_PYTHONPATH")
    for key, value in os.environ.items():
        upper = key.upper()
        if upper.startswith("GIT_"):
            raise BootstrapError("GIT_ENVIRONMENT")
        if upper.startswith("PYTHON") and (
            upper not in allowed_python or value != allowed_python[upper]
        ):
            raise BootstrapError("PYTHON_ENVIRONMENT")
    if any(os.environ.get(key) != value for key, value in allowed_python.items()):
        raise BootstrapError("PYTHON_ENVIRONMENT")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise BootstrapError("GPU_ENVIRONMENT")

    actual_paths = tuple(_path_key(value) for value in sys.path)
    expected_stdlib = tuple(_path_key(value) for value in _STDLIB_PATHS)
    if target["mode"] == "direct":
        if actual_paths != expected_stdlib or _path_key(__file__) != _path_key(
            BOOTSTRAP_WORKTREE_PATH
        ):
            raise BootstrapError("DIRECT_SYS_PATH")
        bootstrap_root = ""
    else:
        python_path = os.environ["PYTHONPATH"]
        if ";" in python_path or os.pathsep in python_path:
            raise BootstrapError("MODULE_PYTHONPATH")
        bootstrap_root = os.path.dirname(os.path.abspath(__file__))
        if not (
            actual_paths == (_path_key(bootstrap_root), *expected_stdlib)
            and _path_key(python_path) == _path_key(bootstrap_root)
            and _path_key(__file__)
            == _path_key(os.path.join(bootstrap_root, "v220b_safe_bootstrap.py"))
            and _path_key(bootstrap_root).startswith(_path_key(RUNTIME_BASE) + "\\")
        ):
            raise BootstrapError("MODULE_SYS_PATH")
    root_key = _path_key(PROJECT_ROOT)
    if any(
        not value
        or value == root_key
        or value.startswith(root_key + "\\")
        or "site-packages" in value
        for value in actual_paths
    ):
        raise BootstrapError("UNSAFE_SYS_PATH")

    cwd_key = _path_key(os.getcwd())
    is_self_check = "--entrypoint-self-check" in target_arguments
    if cwd_key != root_key and not (target["mode"] == "direct" and is_self_check):
        raise BootstrapError("CWD_CONTRACT")
    return {"bootstrap_root": bootstrap_root}


def _load_trusted_stdlib() -> None:
    import importlib as trusted_importlib

    names = (
        "argparse",
        "collections",
        "collections.abc",
        "concurrent.futures",
        "contextlib",
        "copy",
        "ctypes",
        "ctypes.wintypes",
        "dataclasses",
        "enum",
        "functools",
        "hashlib",
        "hmac",
        "importlib.abc",
        "importlib.machinery",
        "importlib.metadata",
        "importlib.util",
        "inspect",
        "io",
        "json",
        "locale",
        "math",
        "ntpath",
        "operator",
        "pathlib",
        "random",
        "re",
        "runpy",
        "shutil",
        "signal",
        "socket",
        "sqlite3",
        "stat",
        "statistics",
        "string",
        "struct",
        "subprocess",
        "tempfile",
        "textwrap",
        "threading",
        "time",
        "traceback",
        "types",
        "typing",
        "unicodedata",
        "uuid",
        "warnings",
        "weakref",
    )
    loaded = {name: trusted_importlib.import_module(name) for name in names}
    globals().update(
        {
            "ctypes": loaded["ctypes"],
            "hashlib": loaded["hashlib"],
            "importlib_abc": loaded["importlib.abc"],
            "importlib_util": loaded["importlib.util"],
            "io": loaded["io"],
            "json": loaded["json"],
            "pathlib": loaded["pathlib"],
            "runpy": loaded["runpy"],
            "sqlite3": loaded["sqlite3"],
            "stat_module": loaded["stat"],
            "tempfile": loaded["tempfile"],
            "types": loaded["types"],
        }
    )
    # pathlib uses io.open; make direct uses obey the same lexical deny even on
    # implementations where an audit event is unavailable.
    if "io.open" not in _ORIGINAL_FUNCTIONS:
        _ORIGINAL_FUNCTIONS["io.open"] = io.open
        io.open = _wrap_single_path(io.open)  # type: ignore[assignment]


def _signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(getattr(value, "st_file_attributes", 0)),
    )


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    """Fields that are stable across Win32 path-stat and descriptor-fstat."""

    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat_module.S_IFMT(value.st_mode)),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(getattr(value, "st_file_attributes", 0)),
    )


def _require_plain_ancestry(path_text: str, *, directory: bool = False) -> os.stat_result:
    path = pathlib.Path(path_text)
    if not path.is_absolute():
        raise BootstrapError("PATH_NOT_ABSOLUTE")
    chain = (path, *path.parents)
    for component in chain:
        observed = os.lstat(component)
        attributes = int(getattr(observed, "st_file_attributes", 0))
        marker = int(getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        if stat_module.S_ISLNK(observed.st_mode) or attributes & marker:
            raise BootstrapError("REPARSE_PATH")
    observed = os.lstat(path)
    expected = stat_module.S_ISDIR if directory else stat_module.S_ISREG
    if not expected(observed.st_mode):
        raise BootstrapError("PATH_TYPE")
    return observed


def _git_blob(payload: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(b"blob " + str(len(payload)).encode("ascii") + b"\0")
    digest.update(payload)
    return digest.hexdigest()


def _read_descriptor(path_text: str) -> tuple[bytes, tuple[int, ...]]:
    lexical = _require_plain_ancestry(path_text)
    flags = os.O_RDONLY
    flags |= int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOINHERIT", 0))
    descriptor = os.open(path_text, flags)
    try:
        before = os.fstat(descriptor)
        if not stat_module.S_ISREG(before.st_mode):
            raise BootstrapError("SOURCE_TYPE")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = os.lstat(path_text)
    # Windows reports different permission bits for path stat (typically 0777)
    # and descriptor fstat (typically 0666).  Compare each observation channel
    # exactly over time, then bind the two channels through stable file identity
    # fields and the regular-file type.
    if (
        _signature(lexical) != _signature(final)
        or _signature(before) != _signature(after)
        or _file_identity(lexical) != _file_identity(before)
        or _file_identity(after) != _file_identity(final)
    ):
        raise BootstrapError("SOURCE_MUTATION")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise BootstrapError("SOURCE_SHORT_READ")
    return payload, _signature(final)


def _read_verified_source(path_text: str, expected_blob: str) -> tuple[bytes, tuple[int, ...]]:
    if not _is_hex_blob(expected_blob):
        raise BootstrapError("BLOB_SHAPE")
    payload, signature = _read_descriptor(path_text)
    normalized = payload.replace(b"\r\n", b"\n")
    if expected_blob.casefold() not in {_git_blob(payload), _git_blob(normalized)}:
        raise BootstrapError("SOURCE_BLOB")
    return payload, signature


def _validate_executable() -> None:
    payload, _signature_value = _read_descriptor(PYTHON_EXECUTABLE)
    if (
        len(payload) != PYTHON_EXECUTABLE_BYTES
        or hashlib.sha256(payload).hexdigest() != PYTHON_EXECUTABLE_SHA256
    ):
        raise BootstrapError("PYTHON_EXECUTABLE")


def _configure_dll_policy() -> None:
    global _DLL_DIRECTORY_HANDLE
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        raise BootstrapError("DLL_PLATFORM")
    _require_plain_ancestry(DLL_DIRECTORY, directory=True)
    handle = os.add_dll_directory(DLL_DIRECTORY)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    setter = kernel32.SetDefaultDllDirectories
    setter.argtypes = [ctypes.wintypes.DWORD]
    setter.restype = ctypes.wintypes.BOOL
    # LOAD_LIBRARY_SEARCH_DEFAULT_DIRS includes application, System32 and
    # AddDllDirectory user directories, and excludes cwd/PATH search.
    if not setter(0x00001000):
        handle.close()
        raise BootstrapError("DLL_POLICY")
    _DLL_DIRECTORY_HANDLE = handle


def _validate_pycache() -> tuple[str, tuple[int, ...]]:
    raw = sys.pycache_prefix
    if not isinstance(raw, str) or not raw:
        raise BootstrapError("PYCACHE_PREFIX")
    if any(part in {".", ".."} for part in raw.replace("/", "\\").split("\\")):
        raise BootstrapError("PYCACHE_LEXICAL")
    key = _path_key(raw)
    base = _path_key(RUNTIME_BASE)
    root = _path_key(PROJECT_ROOT)
    if not key.startswith(base + "\\") or key == base or key.startswith(root + "\\"):
        raise BootstrapError("PYCACHE_LOCATION")
    observed = _require_plain_ancestry(raw, directory=True)
    if os.listdir(raw):
        raise BootstrapError("PYCACHE_NOT_EMPTY")
    return raw.replace("\\", "/"), _signature(observed)


def _revalidate_empty_directory(path_text: str, expected: tuple[int, ...]) -> None:
    observed = _require_plain_ancestry(path_text, directory=True)
    if _signature(observed) != expected or os.listdir(path_text):
        raise BootstrapError("PYCACHE_DRIFT")


class GuardedPath:
    """Read-only sterile sys.path with virtual project-root membership."""

    __slots__ = ("_entries", "_frozen")

    def __init__(self, entries: tuple[str, ...]) -> None:
        object.__setattr__(self, "_entries", tuple(entries))
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise BootstrapError("SYS_PATH_MUTATION")

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __getitem__(self, index):
        value = self._entries[index]
        return tuple(value) if isinstance(index, slice) else value

    def __contains__(self, value: object) -> bool:
        key = _path_key(value)
        return key == _path_key(PROJECT_ROOT) or any(
            key == _path_key(entry) for entry in self._entries
        )

    def __repr__(self) -> str:
        return repr(self._entries)

    def copy(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def count(self, value: object) -> int:
        return int(value in self)

    def index(self, value: object, *args) -> int:
        if _path_key(value) == _path_key(PROJECT_ROOT):
            raise BootstrapError("VIRTUAL_ROOT_NOT_INDEXABLE")
        return self._entries.index(value, *args)

    def _reject(self, *args, **kwargs):
        del args, kwargs
        raise BootstrapError("SYS_PATH_MUTATION")

    append = clear = extend = insert = pop = remove = reverse = sort = _reject
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _reject
    __add__ = __radd__ = __mul__ = __rmul__ = _reject


class _ExactSourceLoader:
    def __init__(self, finder: "_SourceOnlyFinder", fullname: str) -> None:
        self.finder = finder
        self.fullname = fullname

    @property
    def entry(self) -> dict[str, object]:
        return self.finder.entries[self.fullname]

    def is_package(self, fullname: str) -> bool:
        if fullname != self.fullname:
            raise BootstrapError("LOADER_FULLNAME")
        return bool(self.entry["package"])

    def create_module(self, _spec):
        return None

    def get_filename(self, fullname: str) -> str:
        if fullname != self.fullname:
            raise BootstrapError("LOADER_FULLNAME")
        return str(self.entry["path"])

    def get_code(self, fullname: str):
        if fullname != self.fullname:
            raise BootstrapError("LOADER_FULLNAME")
        path_text = str(self.entry["path"])
        payload, signature = _read_verified_source(path_text, str(self.entry["blob"]))
        self.finder.loaded[fullname] = (
            hashlib.sha256(payload).hexdigest(),
            signature,
        )
        return compile(payload, path_text, "exec", dont_inherit=True, optimize=-1)

    def exec_module(self, module) -> None:
        code = self.get_code(self.fullname)
        path_text = str(self.entry["path"])
        package = bool(self.entry["package"])
        module.__file__ = path_text
        module.__loader__ = self
        module.__package__ = self.fullname if package else self.fullname.rpartition(".")[0]
        module.__cached__ = None
        if package:
            module.__path__ = (path_text.rsplit("/", 1)[0],)
        exec(code, module.__dict__)


class _SourceOnlyFinder:
    def __init__(self, target: dict[str, str]) -> None:
        self.entries: dict[str, dict[str, object]] = {}
        for fullname, (relative, blob, package) in _LOCAL_SOURCE_ROWS.items():
            self.entries[fullname] = {
                "path": PROJECT_ROOT + "/" + relative,
                "blob": blob,
                "package": package,
            }
        self.entries[target["module"]] = {
            "path": target["path"],
            "blob": target["blob"],
            "package": False,
        }
        self.loaded: dict[str, tuple[str, tuple[int, ...]]] = {}

    def find_spec(self, fullname: str, _path=None, _target=None):
        if fullname in self.entries:
            loader = _ExactSourceLoader(self, fullname)
            entry = self.entries[fullname]
            spec = importlib_util.spec_from_loader(
                fullname,
                loader,
                origin=str(entry["path"]),
                is_package=bool(entry["package"]),
            )
            if spec is None:
                raise BootstrapError("MODULE_SPEC")
            spec.has_location = True
            spec.cached = None
            if bool(entry["package"]):
                spec.submodule_search_locations[:] = [
                    str(entry["path"]).rsplit("/", 1)[0]
                ]
            return spec
        if fullname == _LOCAL_PREFIXES[0] or any(
            fullname.startswith(prefix + ".") for prefix in _LOCAL_PREFIXES
        ) or fullname in _LOCAL_PREFIXES:
            raise ModuleNotFoundError("local module denied")
        return None

    def revalidate(self) -> None:
        for fullname, (digest, signature) in self.loaded.items():
            entry = self.entries[fullname]
            payload, observed = _read_verified_source(
                str(entry["path"]), str(entry["blob"])
            )
            if hashlib.sha256(payload).hexdigest() != digest or observed != signature:
                raise BootstrapError("IMPORTED_SOURCE_DRIFT")


def _install_source_runtime(target: dict[str, str]) -> tuple[GuardedPath, _SourceOnlyFinder, tuple[object, ...]]:
    for fullname in tuple(sys.modules):
        if fullname in _LOCAL_PREFIXES or fullname.startswith(
            tuple(prefix + "." for prefix in _LOCAL_PREFIXES)
        ):
            raise BootstrapError("LOCAL_MODULE_PRELOADED")
    target_module = target["module"]
    if target_module in sys.modules:
        raise BootstrapError("TARGET_PRELOADED")

    sterile = tuple(path.replace("/", "\\") for path in _STDLIB_PATHS)
    guarded = GuardedPath(sterile)
    sys.path = guarded  # type: ignore[assignment]
    allowed_cache_keys = {_path_key(path) for path in sterile}
    for key in tuple(sys.path_importer_cache):
        if _path_key(key) not in allowed_cache_keys:
            del sys.path_importer_cache[key]
    finder = _SourceOnlyFinder(target)
    previous = tuple(sys.meta_path)
    sys.meta_path.insert(0, finder)
    expected_meta = (finder, *previous)
    return guarded, finder, expected_meta


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BootstrapError("TARGET_JSON_DUPLICATE")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise BootstrapError("TARGET_JSON_NONFINITE")


def _parse_target_receipt(raw: bytes) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise BootstrapError("TARGET_JSON") from error
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise BootstrapError("TARGET_NONCANONICAL")
    return value


def _normalize_exit_code(value: object) -> tuple[int, bool]:
    if value is None:
        return 0, True
    if type(value) is int and 0 <= value <= 255:
        return int(value), True
    return 1, False


def _flush_stream(stream: object) -> None:
    try:
        stream.flush()  # type: ignore[attr-defined]
    except BaseException:
        pass


def _capture_target(action) -> tuple[int, bytes, bytes, bool]:
    _require_plain_ancestry(RUNTIME_BASE, directory=True)
    stdout_temp = tempfile.TemporaryFile(mode="w+b", dir=RUNTIME_BASE)
    stderr_temp = tempfile.TemporaryFile(mode="w+b", dir=RUNTIME_BASE)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_dunder_stdout = sys.__stdout__
    original_dunder_stderr = sys.__stderr__
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    exit_code = 0
    execution_failure = False
    _flush_stream(original_stdout)
    _flush_stream(original_stderr)
    try:
        os.dup2(stdout_temp.fileno(), 1)
        os.dup2(stderr_temp.fileno(), 2)
        try:
            action()
        except SystemExit as error:
            exit_code, valid = _normalize_exit_code(error.code)
            execution_failure = not valid
        except BaseException:
            exit_code = 1
            execution_failure = True
        finally:
            _flush_stream(sys.stdout)
            _flush_stream(sys.stderr)
            _flush_stream(original_stdout)
            _flush_stream(original_stderr)
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        sys.__stdout__ = original_dunder_stdout
        sys.__stderr__ = original_dunder_stderr
    try:
        stdout_temp.seek(0, os.SEEK_END)
        stderr_temp.seek(0, os.SEEK_END)
        stdout_size = stdout_temp.tell()
        stderr_size = stderr_temp.tell()
        if stdout_size > MAX_CAPTURE_BYTES or stderr_size > MAX_CAPTURE_BYTES:
            return 2, b"", b"", True
        stdout_temp.seek(0)
        stderr_temp.seek(0)
        stdout_raw = stdout_temp.read()
        stderr_raw = stderr_temp.read()
    finally:
        stdout_temp.close()
        stderr_temp.close()
    return exit_code, stdout_raw, stderr_raw, execution_failure


def _execute_direct(path_text: str, payload: bytes, target_arguments: tuple[str, ...]) -> None:
    code = compile(payload, path_text, "exec", dont_inherit=True, optimize=-1)
    target_module = types.ModuleType("__main__")
    target_module.__dict__.update(
        {
            "__builtins__": builtins,
            "__file__": path_text,
            "__loader__": None,
            "__name__": "__main__",
            "__package__": None,
            "__spec__": None,
        }
    )
    previous_main = sys.modules.get("__main__")
    previous_argv = sys.argv
    sys.modules["__main__"] = target_module
    sys.argv = [path_text, *target_arguments]
    try:
        exec(code, target_module.__dict__)
    finally:
        sys.argv = previous_argv
        if previous_main is None:
            del sys.modules["__main__"]
        else:
            sys.modules["__main__"] = previous_main


def _execute_module(fullname: str, path_text: str, target_arguments: tuple[str, ...]) -> None:
    previous_argv = sys.argv
    sys.argv = [path_text, *target_arguments]
    try:
        runpy.run_module(fullname, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = previous_argv


def _attestation(target: dict[str, str], bootstrap_blob: str, pycache_prefix: str):
    value = {
        "mode": target["mode"],
        "bootstrap_blob": bootstrap_blob.casefold(),
        "target_blob": target["blob"].casefold(),
        "source_only": True,
        "guarded_path": True,
        "pycache_prefix": pycache_prefix,
    }
    # The attribute name and exact six-key MappingProxyType schema are part of
    # the runner/bootstrap interface.
    proxy = types.MappingProxyType(value)
    setattr(sys, ATTESTATION_ATTRIBUTE, proxy)
    return proxy


def _empty_attestation() -> dict[str, object]:
    return {
        "mode": "",
        "bootstrap_blob": "",
        "target_blob": "",
        "source_only": False,
        "guarded_path": False,
        "pycache_prefix": "",
    }


def _write_outer(envelope: dict[str, object]) -> None:
    payload = _canonical_bytes(envelope) + b"\n"
    view = memoryview(payload)
    while view:
        written = os.write(1, view)
        if written <= 0:
            raise BootstrapError("OUTER_WRITE")
        view = view[written:]


def _bootstrap_main(raw_arguments: list[str]) -> int:
    global _CURRENT_ATTESTATION
    _install_lexical_guard()
    parsed, target_arguments = _parse_cli(raw_arguments)
    target = _match_target(parsed)
    bootstrap_blob = parsed["--bootstrap-blob"].casefold()
    if not _is_hex_blob(bootstrap_blob):
        raise BootstrapError("BOOTSTRAP_BLOB_SHAPE")
    startup = _validate_early_runtime(target, target_arguments)

    # No repository root was reachable during these imports.
    _load_trusted_stdlib()
    _configure_dll_policy()
    _validate_executable()
    if sqlite3.sqlite_version != SQLITE_VERSION:
        raise BootstrapError("SQLITE_RUNTIME")
    pycache_prefix, pycache_signature = _validate_pycache()

    if target["mode"] == "module":
        bootstrap_root = startup["bootstrap_root"]
        if os.listdir(bootstrap_root) != ["v220b_safe_bootstrap.py"]:
            raise BootstrapError("MODULE_ROOT_ENTRIES")
    own_path = os.path.abspath(__file__).replace("\\", "/")
    own_payload, own_signature = _read_verified_source(own_path, bootstrap_blob)
    own_digest = hashlib.sha256(own_payload).hexdigest()

    guarded, finder, expected_meta = _install_source_runtime(target)
    attestation_proxy = _attestation(target, bootstrap_blob, pycache_prefix)
    _CURRENT_ATTESTATION = dict(attestation_proxy)

    direct_identity: tuple[str, tuple[int, ...]] | None = None
    if target["mode"] == "direct":
        target_payload, target_signature = _read_verified_source(
            target["path"], target["blob"]
        )
        direct_identity = (
            hashlib.sha256(target_payload).hexdigest(),
            target_signature,
        )
        action = lambda: _execute_direct(
            target["path"], target_payload, target_arguments
        )
    else:
        action = lambda: _execute_module(
            target["module"], target["path"], target_arguments
        )

    target_exit_code, target_stdout, target_stderr, execution_failure = _capture_target(
        action
    )

    # Revalidate all execution boundaries before emitting the only observable
    # receipt.  The compile operation already used the bytes read and verified
    # through the same descriptor.
    if sys.path is not guarded or tuple(sys.meta_path) != expected_meta:
        raise BootstrapError("IMPORT_RUNTIME_MUTATION")
    if getattr(sys, ATTESTATION_ATTRIBUTE, None) is not attestation_proxy:
        raise BootstrapError("ATTESTATION_MUTATION")
    finder.revalidate()
    if direct_identity is not None:
        payload, signature = _read_verified_source(target["path"], target["blob"])
        if (
            hashlib.sha256(payload).hexdigest() != direct_identity[0]
            or signature != direct_identity[1]
        ):
            raise BootstrapError("TARGET_SOURCE_DRIFT")
    final_own, final_own_signature = _read_verified_source(own_path, bootstrap_blob)
    if hashlib.sha256(final_own).hexdigest() != own_digest or final_own_signature != own_signature:
        raise BootstrapError("BOOTSTRAP_SOURCE_DRIFT")
    _revalidate_empty_directory(pycache_prefix, pycache_signature)

    target_receipt: dict[str, object] | None
    if target_stderr or execution_failure:
        target_receipt = None
        target_exit_code = 2
    elif not target_stdout:
        if target_exit_code == 0:
            target_exit_code = 2
        target_receipt = None
    else:
        try:
            target_receipt = _parse_target_receipt(target_stdout)
        except BootstrapError:
            target_receipt = None
            target_exit_code = 2

    envelope = {
        "bootstrap": dict(attestation_proxy),
        "target_exit_code": target_exit_code,
        "target_receipt": target_receipt,
    }
    _write_outer(envelope)
    return target_exit_code


def _write_minimal_failure() -> None:
    # This literal is canonical JSON and requires no filesystem-backed import;
    # it remains available if startup validation fails before json is trusted.
    payload = (
        b'{"bootstrap":{"bootstrap_blob":"","guarded_path":false,'
        b'"mode":"","pycache_prefix":"","source_only":false,'
        b'"target_blob":""},"target_exit_code":2,"target_receipt":null}\n'
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(1, view)
            if written <= 0:
                break
            view = view[written:]
    except BaseException:
        pass


def main(argv: list[str] | None = None) -> int:
    try:
        return _bootstrap_main(list(sys.argv[1:] if argv is None else argv))
    except BaseException:
        # Do not disclose paths, exception messages, raw stderr, or traceback.
        _write_minimal_failure()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

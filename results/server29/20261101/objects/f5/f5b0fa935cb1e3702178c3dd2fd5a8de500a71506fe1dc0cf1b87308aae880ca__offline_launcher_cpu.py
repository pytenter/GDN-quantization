"""Execute the production launcher in an isolated CPU virtual-path fixture.

Only external admission/NVML/process boundaries are mocked. No production
path is created and no tmux, GPU API, model, worker, or finisher is invoked.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import types


TASK = 'QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1'
ROOT = PurePosixPath('/tmp') / TASK


class Fixture:
    def __init__(self, source):
        self.tmp = tempfile.TemporaryDirectory(prefix='wscale-launcher-cpu-')
        self.base = Path(self.tmp.name)
        self.calls = []
        self.admission_calls = []
        self.fail_device = None
        self.fail_jobs = False
        self.fail_model = False
        self.fail_storage = False
        self.has_session_rc = 1
        self.new_session_rc = 0
        self.pane_rc = 0
        fixture = self

        class VirtualPath:
            def __init__(self, value):
                self.value = value.value if isinstance(value, VirtualPath) else PurePosixPath(str(value))
            def __truediv__(self, other): return VirtualPath(self.value / str(other))
            def __str__(self): return str(self.value)
            def __fspath__(self): return str(self.value)
            def __eq__(self, other): return isinstance(other, VirtualPath) and self.value == other.value
            def resolve(self): return self
            @property
            def parents(self): return tuple(VirtualPath(v) for v in self.value.parents)
            def actual(self):
                # No absolute virtual path can escape the temporary fixture.
                return fixture.base / str(self.value).lstrip('/')
            def exists(self): return self.actual().exists()
            def glob(self, pattern):
                for p in self.actual().glob(pattern):
                    yield VirtualPath('/' + str(p.relative_to(fixture.base)))

        self.Path = VirtualPath
        self.root = VirtualPath(ROOT)
        self.C = types.ModuleType('execution_control')
        C = self.C
        C.TASK = TASK
        C.AUTHORIZED_GPUS = (1, 2, 3)
        def require(value, message):
            if not value: raise RuntimeError(message)
        C.require = require
        C.read = lambda p: json.loads(p.actual().read_text())
        C.sha = lambda p: hashlib.sha256(p.actual().read_bytes()).hexdigest()
        def atomic_new(p, value):
            q = p.actual(); q.parent.mkdir(parents=True, exist_ok=True)
            with q.open('x') as f: json.dump(value, f, sort_keys=True)
        C.atomic_new = atomic_new
        C.process_identity = lambda pid: dict(pid=pid, starttick=12345,
            cmdline=['CPU_FIXTURE_ONLY', str(pid)])
        def authorize(root, permit):
            self.admission_calls.append('authorize')
            p = C.read(permit)
            require(p['source_freeze_sha256'] == C.sha(root/'SOURCE_FREEZE.json'),
                'CPU_FIXTURE permit/source pin failure')
            for row in C.read(root/'SOURCE_FREEZE.json')['files']:
                q = root / row['path']
                require(q.actual().stat().st_size == row['bytes'] and C.sha(q) == row['sha256'],
                    'CPU_FIXTURE source byte failure')
            require(p['ALLOW_GPU'] == 'YES', 'CPU_FIXTURE permit denied')
            return p
        C.authorize = authorize
        def model_guard(root):
            self.admission_calls.append('model_guard')
            require(not self.fail_model, 'CPU_FIXTURE model guard failure')
        C.model_guard = model_guard
        def storage_guard(root):
            self.admission_calls.append('storage_guard')
            require(not self.fail_storage, 'CPU_FIXTURE storage guard failure')
            return dict(free_bytes=10*1024**3, shared_data_free_bytes=9*1024**3)
        C.storage_guard = storage_guard
        def validate_jobs(root, jobs):
            self.admission_calls.append('validate_jobs')
            require(not self.fail_jobs and jobs == ['CPU_FIXTURE_90_JOB_MANIFEST'],
                'CPU_FIXTURE formal dispatch validation failure')
        C.validate_jobs = validate_jobs
        def device(gpu):
            self.admission_calls.append(('device', gpu))
            require(gpu != self.fail_device, 'CPU_FIXTURE wrong UUID or foreign GPU owner')
            return dict(physical_index=gpu, uuid='CPU_FIXTURE_UUID_'+str(gpu),
                observation='MOCK_ONLY_NOT_GPU_PROOF')
        C.device = device
        def run(argv, **kwargs):
            self.calls.append(list(argv))
            require(argv[0] == 'tmux', 'Unexpected executable in fixture')
            if argv[1] == 'has-session':
                return types.SimpleNamespace(returncode=self.has_session_rc, stdout='', stderr='')
            if argv[1] == 'new-session':
                return types.SimpleNamespace(returncode=self.new_session_rc, stdout='', stderr='mock launch error' if self.new_session_rc else '')
            if argv[1] == 'display-message':
                return types.SimpleNamespace(returncode=self.pane_rc, stdout='76543\n', stderr='')
            raise AssertionError('Unexpected tmux operation')
        namespace = dict(__name__='cpu_fixture_production_launcher',
            __file__=str(ROOT/'operations/launch_offline_once.py'))
        exec(compile(source, str(ROOT/'operations/launch_offline_once.py'), 'exec'), namespace)
        namespace['Path'] = VirtualPath
        namespace['subprocess'] = types.SimpleNamespace(run=run)
        self.main = namespace['main']
        self.write('src/frozen_boundary.py', 'CPU fixture source\n', raw=True)
        raw = (self.root/'src/frozen_boundary.py').actual().read_bytes()
        self.write('SOURCE_FREEZE.json', dict(files=[dict(path='src/frozen_boundary.py',
            bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())]))
        self.write('GPU_PERMIT.json', dict(ALLOW_GPU='YES',
            source_freeze_sha256=C.sha(self.root/'SOURCE_FREEZE.json')))
        self.write('CPU_CODEC_QUALIFICATION.json', dict(status='PASS_CPU_CONTRACTS_AND_INTEGRATION_NOT_GPU'))
        self.write('calibration/FP_COLLECTION_EXIT_RELEASE.json', dict(status='PASS_ACTUAL_EXIT0_AND_RELEASE'))
        self.write('calibration/ARCHIVE29_VERIFICATION.json', dict(status='PASS_INDEPENDENT29_NONAIME_RAW_SOURCE_SHA_BYTES_NOREPLACE'))
        self.write('FORMAL_JOB_MANIFEST.json', dict(jobs=['CPU_FIXTURE_90_JOB_MANIFEST']))

    def write(self, name, value, raw=False):
        p = (self.root/name).actual(); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value if raw else json.dumps(value, sort_keys=True))

    def launch(self, root=None, cvd=''):
        previous = sys.modules.get('execution_control')
        prior_path = list(sys.path)
        old_cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
        sys.modules['execution_control'] = self.C
        os.environ['CUDA_VISIBLE_DEVICES'] = cvd
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.main(self.root if root is None else root)
        finally:
            sys.path[:] = prior_path
            if previous is None: sys.modules.pop('execution_control', None)
            else: sys.modules['execution_control'] = previous
            if old_cvd is None: os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else: os.environ['CUDA_VISIBLE_DEVICES'] = old_cvd

    def new_calls(self): return [x for x in self.calls if x[1] == 'new-session']
    def read(self, name): return self.C.read(self.root/name)
    def close(self): self.tmp.cleanup()


def run(source):
    rows = []
    def case(name, prepare=None, success=False, check=None):
        f = Fixture(source)
        try:
            if prepare: prepare(f)
            error = None
            try: f.launch()
            except (RuntimeError, FileExistsError, FileNotFoundError) as exc: error = str(exc)
            assert (error is None) == success, (name, error)
            if not success: assert not f.new_calls(), (name, f.calls)
            if check: check(f)
            rows.append(dict(name=name, status='PASS', blocked=not success,
                tmux_new_session_calls=len(f.new_calls()), expected_error=error))
        finally: f.close()
    case('source bytes changed blocks', lambda f:f.write('src/frozen_boundary.py', 'changed', raw=True))
    case('permit source SHA changed blocks', lambda f:f.write('GPU_PERMIT.json', dict(ALLOW_GPU='YES',source_freeze_sha256='0'*64)))
    case('permit denied blocks', lambda f:f.write('GPU_PERMIT.json', dict(ALLOW_GPU='NO',source_freeze_sha256=f.C.sha(f.root/'SOURCE_FREEZE.json'))))
    case('model stat boundary failure blocks', lambda f:setattr(f,'fail_model',True))
    case('storage boundary failure blocks', lambda f:setattr(f,'fail_storage',True))
    case('CPU qualification failure blocks', lambda f:f.write('CPU_CODEC_QUALIFICATION.json',dict(status='FAIL')))
    case('calibration exit receipt failure blocks', lambda f:f.write('calibration/FP_COLLECTION_EXIT_RELEASE.json',dict(status='FAIL')))
    case('calibration durable archive failure blocks', lambda f:f.write('calibration/ARCHIVE29_VERIFICATION.json',dict(status='FAIL')))
    case('formal manifest validation failure blocks', lambda f:setattr(f,'fail_jobs',True))
    for g in (1,2,3):
        case('device UUID or foreign owner GPU'+str(g)+' blocks', lambda f,g=g:setattr(f,'fail_device',g))
    case('existing run started blocks', lambda f:f.write('GPU_RUN_STARTED.json',{}))
    case('existing terminal session result blocks', lambda f:f.write('SESSION_RESULT.json',{}))
    case('existing qualification dispatch blocks', lambda f:f.write('execution/unit/DISPATCH.json',{}))
    case('existing formal dispatch blocks', lambda f:f.write('formal/unit/DISPATCH.json',{}))
    case('existing tmux session blocks', lambda f:setattr(f,'has_session_rc',0))
    case('unresolved tmux query blocks', lambda f:setattr(f,'has_session_rc',2))
    case('existing finisher output blocks', lambda f:f.write('finisher.log','old output',raw=True))
    case('existing attempt blocks without truncation', lambda f:f.write('operations/OFFLINE_LAUNCH_ATTEMPT.json',dict(original='preserved')),
        check=lambda f:assert_equal(f.read('operations/OFFLINE_LAUNCH_ATTEMPT.json'),dict(original='preserved')))
    case('existing admission blocks duplicate launcher', lambda f:f.write('operations/OFFLINE_LAUNCH_CPU_ADMISSION.json',dict(original='preserved')),
        check=lambda f:assert_equal(f.read('operations/OFFLINE_LAUNCH_CPU_ADMISSION.json'),dict(original='preserved')))
    # Root is verified before any C admission, device query, or subprocess.
    f=Fixture(source)
    try:
        try: f.launch(f.Path('/tmp/'+TASK+'_WRONG'))
        except RuntimeError: pass
        else: raise AssertionError('Wrong root accepted')
        assert not f.admission_calls and not f.calls
        rows.append(dict(name='wrong absolute root blocks before admission',status='PASS',tmux_new_session_calls=0))
    finally:f.close()
    f=Fixture(source)
    try:
        try:f.launch(cvd='CPU_FIXTURE_NONEMPTY')
        except RuntimeError:pass
        else:raise AssertionError('GPU-visible launcher accepted')
        assert not f.admission_calls and not f.calls
        rows.append(dict(name='nonempty CUDA visibility blocks before admission',status='PASS',tmux_new_session_calls=0))
    finally:f.close()
    # Successful dispatch is one tmux launch, CPU-only argv, real receipt path.
    f=Fixture(source)
    try:
        f.launch()
        assert len(f.new_calls())==1
        assert [x for x in f.admission_calls if isinstance(x,tuple)]==[('device',1),('device',2),('device',3)]
        receipt=f.read('operations/OFFLINE_LAUNCH_RECEIPT.json')
        assert receipt['actual_returncode']==0 and receipt['pane_query_returncode']==0
        assert receipt['actual_offline_shell_identity']['pid']==76543
        assert receipt['argv'][:2]==['env','CUDA_VISIBLE_DEVICES=']
        assert f.read('operations/OFFLINE_LAUNCH_ATTEMPT.json')['no_automatic_retry'] is True
        prior_attempt=(f.root/'operations/OFFLINE_LAUNCH_ATTEMPT.json').actual().read_bytes()
        try:f.launch()
        except (RuntimeError,FileExistsError):pass
        else:raise AssertionError('Duplicate launch accepted')
        assert len(f.new_calls())==1
        assert prior_attempt==(f.root/'operations/OFFLINE_LAUNCH_ATTEMPT.json').actual().read_bytes()
        rows.append(dict(name='success one launch and repeat blocked',status='PASS',tmux_new_session_calls=1))
    finally:f.close()
    # A failed tmux launch is preserved and cannot be retried automatically.
    f=Fixture(source)
    try:
        f.new_session_rc=7
        try:f.launch()
        except RuntimeError:pass
        else:raise AssertionError('Failed tmux launch accepted')
        assert len(f.new_calls())==1
        assert f.read('operations/OFFLINE_LAUNCH_RECEIPT.json')['actual_returncode']==7
        try:f.launch()
        except (RuntimeError,FileExistsError):pass
        else:raise AssertionError('Failed attempt was retried')
        assert len(f.new_calls())==1
        rows.append(dict(name='tmux failure preserved and repeat blocked',status='PASS',tmux_new_session_calls=1))
    finally:f.close()
    # Pane query failure must not create a second session.
    case('pane observation failure retains successful launch receipt',lambda f:setattr(f,'pane_rc',1),success=True,
        check=lambda f:assert_equal(f.read('operations/OFFLINE_LAUNCH_RECEIPT.json')['pane_query_returncode'],1))
    return dict(status='PASS_CPU_ACTUAL_LAUNCHER_ADMISSION_AND_ONCE_ONLY_SIDE_EFFECTS',
        production_source_sha256=hashlib.sha256(source.encode()).hexdigest(),checks=len(rows),rows=rows,
        GPU_calls=0,model_loads=0,actual_tmux_calls=0,actual_worker_or_finisher_calls=0,
        fixture_root='Temporary physical directory with virtual exact /tmp/TASK paths',
        boundaries='C source/permit/model/storage/jobs/device and subprocess mocked; this is CPU control proof, not GPU qualification')


def assert_equal(actual, expected): assert actual==expected,(actual,expected)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    print(json.dumps(run((root/'operations/launch_offline_once.py').read_text()),sort_keys=True))

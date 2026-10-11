"""Migration contracts: isolate consumer writes, preserve bytes, reject stale inputs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .godot_delivery import checked_payloads, deploy_pack
from .runner import file_hash, write_json
from tools.reference_snapshot import verify_reference


class IndependentDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root/'producer/assets/generated/als_nonprop_example'
        self.source.mkdir(parents=True)
        (self.source/'raw.json').write_bytes(b'{"sample": 1}\r\n')
        self.pack = self.source/'resource-pack.json'
        self.consumer = self.root/'consumer'
        self.consumer.mkdir()
        (self.consumer/'project.godot').write_text('config_version=5\n')
        write_json(self.pack, dict(files=[dict(path='res://assets/generated/als_nonprop_example/raw.json',
                                              sha256=file_hash(self.source/'raw.json'))]))

    def tearDown(self):
        self.temp.cleanup()

    def test_deployment_preserves_bytes_and_source_and_refuses_overwrite(self):
        original = self.pack.read_bytes()
        deployed = deploy_pack(self.pack, self.consumer, self.root/'evidence')
        self.assertEqual(original, deployed.read_bytes())
        self.assertEqual(original, self.pack.read_bytes())
        self.assertEqual((self.source/'raw.json').read_bytes(), (deployed.parent/'raw.json').read_bytes())
        self.assertFalse((self.consumer/'assets/config/als_resource_pack.json').exists())
        with self.assertRaisesRegex(ValueError, 'already exists'):
            deploy_pack(self.pack, self.consumer, self.root/'second')

    def test_changed_payload_blocks_deployment(self):
        (self.source/'raw.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed'):
            deploy_pack(self.pack, self.consumer, self.root/'evidence')
        self.assertFalse((self.consumer/'assets').exists())

    def test_escaping_payload_is_rejected(self):
        outside = self.root/'outside.json'
        outside.write_text('{}')
        write_json(self.pack, dict(files=[dict(path='res://../../../outside.json', sha256=file_hash(outside))]))
        with self.assertRaisesRegex(ValueError, 'escapes'):
            checked_payloads(self.pack)

    def test_reference_tampering_is_rejected(self):
        reference = self.root/'reference'
        reference.mkdir()
        payload = reference/'raw.json'
        payload.write_bytes(b'original')
        write_json(reference/'reference.json', dict(files=[dict(path='raw.json', sha256=file_hash(payload))], bytes=8))
        self.assertTrue(verify_reference(reference)['passed'])
        payload.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            verify_reference(reference)

    def test_wsl_handoff_translates_paths_and_keeps_runtime_failure(self):
        from .godot_delivery import deploy_and_verify
        runtime_python = self.root/'python.exe'
        runtime_python.touch()
        output = self.root/'runtime'

        def run(command, **kwargs):
            self.assertTrue(all(not str(value).startswith('/mnt/') for value in command[1:]))
            self.assertIn('--runtime-case', command)
            write_json(output/'runtime-result.json', dict(status='runtime_failed', checks=[dict(passed=False)]))
            return type('Process', (), dict(returncode=1))()

        with patch('tools.kimodo_pipeline.godot_delivery.project_root', return_value=self.consumer), \
             patch('tools.kimodo_pipeline.godot_delivery.configured_path', return_value=runtime_python), \
             patch('tools.kimodo_pipeline.godot_delivery.requires_windows_bridge', return_value=True), \
             patch('tools.kimodo_pipeline.godot_delivery.windows_path', side_effect=lambda p:'Z:\\'+Path(p).name), \
             patch('tools.kimodo_pipeline.godot_delivery.subprocess.run', side_effect=run):
            result = deploy_and_verify(self.pack, output, self.root/'godot.exe', self.root/'dotnet.exe', runtime_python,
                                       runtime_cases=['walk-forward'])
        self.assertEqual('runtime_failed', result['status'])
        self.assertEqual(file_hash(self.pack), result['sourcePackSha256'])

    def test_wsl_metadata_paths_change_without_rewriting_motion_or_ue_identity(self):
        from .godot_delivery import adapt_windows_metadata
        native=self.root/'native.json'
        write_json(native,dict(authoringBundle='/mnt/f/work/authoring.json',jobs=[dict(
            source='/ALS/Animations/Walk.Walk',raw='/mnt/f/work/retarget.raw.json',
            motionPath='/mnt/f/work/motion.npz',rawSha256='preserved')]))
        # Force the POSIX-source branch while keeping the local fixture readable.
        original=json.loads(self.pack.read_text(encoding='utf-8'))
        original['nativeRetargetBundle']=native.as_posix()
        write_json(self.pack,original)
        destination=deploy_pack(self.pack,self.consumer,self.root/'evidence')
        before=(self.source/'raw.json').read_bytes()
        with patch('tools.kimodo_pipeline.godot_delivery.PurePosixPath') as posix:
            posix.return_value.is_absolute.return_value=True
            adapt_windows_metadata(destination,self.root/'evidence',lambda p:'F:\\translated\\'+Path(str(p)).name)
        saved=json.loads((destination.parent/'deployment/retarget-bundle.json').read_text(encoding='utf-8'))
        self.assertEqual('/ALS/Animations/Walk.Walk',saved['jobs'][0]['source'])
        self.assertTrue(saved['jobs'][0]['raw'].startswith('F:\\translated\\'))
        self.assertEqual('preserved',saved['jobs'][0]['rawSha256'])
        self.assertEqual(before,(destination.parent/'raw.json').read_bytes())
        self.assertEqual(before,(self.source/'raw.json').read_bytes())
        checked_payloads(destination)


if __name__ == '__main__':
    unittest.main()

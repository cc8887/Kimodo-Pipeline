"""Finalization must not certify subsets, stale payloads or failed FBX reads."""
import hashlib
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from .offline_pipeline import finalize_staged_pack
from .runner import write_json, file_hash
from .graph import read_json
from .publish import runtime_asset_integrity_files


class FinalizerIntegrityTests(unittest.TestCase):
    def setup_pack(self, root):
        pack=root/'assets/generated/als_nonprop_test';pack.mkdir(parents=True)
        reference=root/'reference';config=reference/'assets/config';config.mkdir(parents=True)
        write_json(config/'v4_recovery_movement_source_inputs.json',{'skeletons':[{}]})
        write_json(config/'refactored_animation_sources.json',{'skeletons':{'/ALS/ALS/Character/SK_Als.SK_Als':{}}})
        original=reference/'assets/generated/als_v4';original.mkdir(parents=True)
        mesh=original/'mesh.fbx';mesh.write_bytes(b'mesh')
        write_json(original/'als_manifest.json',dict(skeletalMeshes=[dict(assetName='Mannequin',outputPath='mesh.fbx')],
            files=[dict(relativePath='mesh.fbx',sha256=file_hash(mesh))]))
        source='test_animation';folder=pack/'root-ownership'/hashlib.sha256(source.encode()).hexdigest();folder.mkdir(parents=True)
        write_json(folder/'runtime.raw.json',dict(source=source,tracks=[]))
        write_json(folder/'ownership.json',dict(source=source))
        fbx=pack/'v4/animation.fbx';fbx.parent.mkdir();fbx.write_bytes(b'initial-fbx')
        manifest=dict(animations=[dict(objectPath=source,outputPath='animation.fbx',kimodoSource={})],
            files=[dict(relativePath='animation.fbx',sha256=file_hash(fbx),size=fbx.stat().st_size)])
        write_json(pack/'v4/als_manifest.json',manifest)
        control=pack/'control.json';write_json(control,dict(untouched=True))
        jobs=[dict(source=source,edition='v4')];native=root/'retarget.json';write_json(native,dict(jobs=jobs))
        prefix='res://'+pack.relative_to(root).as_posix()
        payloads=[folder/'runtime.raw.json',folder/'ownership.json',fbx,pack/'v4/als_manifest.json',control]
        files=[dict(path='res://'+p.relative_to(root).as_posix(),sha256=file_hash(p)) for p in payloads]
        staged=dict(files=files,assetRoot=prefix+'/v4',nativeRetargetBundle=str(native),replacedAnimations=1,
            rootOwnershipReceipts=[dict(source=source,file='root-ownership/'+folder.name+'/ownership.json')])
        staged['runtimeAssetIntegrityFiles']=runtime_asset_integrity_files(files,staged['assetRoot'],manifest)
        write_json(pack/'resource-pack.json',staged)
        output=root/'output';output.mkdir()
        return pack,output,jobs,reference,config,control

    def mocks(self, root, reference, config, *, passed=True, mutate=None):
        stack=ExitStack()
        for name,value in [('ROOT',root),('CONFIG',config)]:
            stack.enter_context(patch('tools.kimodo_pipeline.offline_pipeline.'+name,value))
        stack.enter_context(patch('tools.kimodo_pipeline.offline_pipeline.reference_root',return_value=reference))
        stack.enter_context(patch('tools.kimodo_pipeline.offline_pipeline.mesh_binding_skeleton',side_effect=lambda skel,path:skel))
        stack.enter_context(patch('tools.kimodo_pipeline.offline_pipeline.write_fbx',side_effect=lambda path,*a:path.write_bytes(b'final-fbx')))
        def readback(*args):
            if mutate:mutate()
            return dict(passed=passed)
        stack.enter_context(patch('tools.kimodo_pipeline.offline_pipeline.readback',side_effect=readback))
        viewer=stack.enter_context(patch('tools.kimodo_pipeline.viewer_pack.export_pack',return_value={'id':'test'}))
        return stack,viewer

    def test_final_hashes_bind_fbx_and_runtime_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();pack,out,jobs,ref,config,_=self.setup_pack(root)
            stack,_=self.mocks(root,ref,config)
            with stack:staged,_=finalize_staged_pack(jobs,pack,out)
            expected=file_hash(pack/'v4/animation.fbx')
            self.assertEqual(expected,next(e['sha256'] for e in staged['files'] if e['path'].endswith('/animation.fbx')))
            self.assertEqual(expected,next(e['sha256'] for e in staged['runtimeAssetIntegrityFiles'] if e['path'].endswith('/animation.fbx')))
            self.assertTrue(staged['offlineFbxReadback']['passed'])

    def test_empty_and_duplicate_jobs_cannot_be_certified(self):
        for kind in ('empty','duplicate'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                root=Path(directory).resolve();pack,out,jobs,ref,config,_=self.setup_pack(root)
                stack,_=self.mocks(root,ref,config)
                with stack,self.assertRaisesRegex(ValueError,'exact staged source closure'):
                    finalize_staged_pack([] if kind=='empty' else jobs*2,pack,out)

    def test_failed_readback_saves_failure_and_never_registers_successful_viewer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();pack,out,jobs,ref,config,_=self.setup_pack(root)
            stack,viewer=self.mocks(root,ref,config,passed=False)
            with stack,self.assertRaisesRegex(ValueError,'Final FBX readback failed'):
                finalize_staged_pack(jobs,pack,out)
            self.assertFalse(read_json(pack/'resource-pack.json')['offlineFbxReadback']['passed'])
            self.assertEqual(['test_animation'],read_json(out/'final-fbx-progress.json')['failed'])
            viewer.assert_not_called()

    def test_unrelated_payload_change_is_rejected_before_and_during_export(self):
        for when in ('before','during'):
            with self.subTest(when=when),tempfile.TemporaryDirectory() as directory:
                root=Path(directory).resolve();pack,out,jobs,ref,config,control=self.setup_pack(root)
                change=lambda:write_json(control,dict(untouched=False))
                if when=='before':change()
                stack,_=self.mocks(root,ref,config,mutate=change if when=='during' else None)
                with stack,self.assertRaisesRegex(ValueError,'payload changed'):
                    finalize_staged_pack(jobs,pack,out)

from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import digest,file_hash
from snapshot import write_archive,load_archive


def test_snapshot_roundtrip_preserves_float32_bits_and_unicode_ids(tmp_path):
    vector=np.zeros((1,1024),dtype=np.float32);vector[0,:3]=[.6,.8,-0.0]
    arrays={'documents_A_metadata':vector,'official_query_vectors':vector.copy(),
            'panel_query_vectors':vector.copy(),'skill_ids':np.asarray(['public-id-中文'])}
    first=tmp_path/'a.npz';second=tmp_path/'b.npz'
    write_archive(first,arrays);write_archive(second,arrays)
    assert file_hash(first)==file_hash(second)
    manifest={'sha256':file_hash(first),'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype),
                'sha256_contiguous_bytes':digest(v.tobytes())} for k,v in arrays.items()}}
    output=load_archive(first,manifest)
    assert all(output[k].tobytes()==v.tobytes() for k,v in arrays.items())
    assert np.signbit(output['documents_A_metadata'][0,2])
    first.write_bytes(first.read_bytes()+b'corrupt')
    with pytest.raises(AssertionError,match='corrupt archive'):load_archive(first,manifest)


def test_snapshot_rejects_object_pickle_payload(tmp_path):
    with pytest.raises(AssertionError):write_archive(tmp_path/'unsafe.npz',{'bad':np.array([{'x':1}],dtype=object)})

import storage


def test_project_dir_creates_the_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "data_dir", lambda: tmp_path)
    path = storage.project_dir("/some/repo", "rag_index")
    assert path.is_dir()


def test_project_dir_path_does_not_create_anything(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "data_dir", lambda: tmp_path)
    path = storage.project_dir_path("/some/repo", "rag_index")
    assert not path.exists()
    assert not path.parent.exists()


def test_project_dir_path_matches_project_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "data_dir", lambda: tmp_path)
    assert storage.project_dir_path("/some/repo", "rag_index") == storage.project_dir("/some/repo", "rag_index")

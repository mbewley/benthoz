import pandas as pd
import pytest

import prep.image_fetcher

# A real row from BENTHOZ-2015-imagelist.csv.
WEB_LOCATION = "PS201012/r20101217_021510_fingal_07_dense_fihpz_b/images/PR_20101217_025403_165_LC16.png"
EXPECTED_URL = (
    "https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/AUV/"
    "PS201012/r20101217_021510_fingal_07_dense_fihpz_b/"
    "i20101217_021510_gtif/PR_20101217_025403_165_LC16.tif"
)


class TestWebLocationToUrl:
    def test_known_location(self):
        assert prep.image_fetcher.web_location_to_url(WEB_LOCATION) == EXPECTED_URL

    def test_png_only_replaced_in_the_extension(self):
        """A dive name containing 'png' must survive the extension rewrite."""
        web_location = "PS201012/r20101217_021510_pngbay_01/images/PR_20101217_025403_165_LC16.png"
        url = prep.image_fetcher.web_location_to_url(web_location)
        assert url.endswith("/PR_20101217_025403_165_LC16.tif")
        assert "pngbay" in url

    def test_every_row_in_the_real_manifest_converts(self, repo_root):
        """Structural invariants across all 9,882 published images."""
        manifest = repo_root / "BENTHOZ-2015-imagelist.csv"
        df = pd.read_csv(manifest).query('image__id != "image__id"')

        urls = df.web_location.map(prep.image_fetcher.web_location_to_url)

        assert len(urls) == len(df)
        assert urls.str.startswith(prep.image_fetcher.S3_ROOT).all()
        assert urls.str.endswith(".tif").all()
        assert not urls.str.contains("/images/").any()
        assert urls.str.contains("_gtif/").all()


class TestFetchImage:
    def test_skips_existing_file(self, tmp_path):
        """An already-downloaded image must not be re-fetched."""
        existing = tmp_path / "PR_20101217_025403_165_LC16.tif"
        existing.write_bytes(b"already here")

        class ExplodingSession:
            def get(self, url):
                raise AssertionError("should not have been called")

        path = prep.image_fetcher.fetch_image(EXPECTED_URL, str(tmp_path), ExplodingSession())

        assert path == str(existing)
        assert existing.read_bytes() == b"already here"

    def test_writes_fetched_content(self, tmp_path):
        class FakeResponse:
            ok = True
            content = b"tiff bytes"

        class FakeSession:
            def get(self, url):
                return FakeResponse()

        path = prep.image_fetcher.fetch_image(EXPECTED_URL, str(tmp_path), FakeSession())

        assert (tmp_path / "PR_20101217_025403_165_LC16.tif").read_bytes() == b"tiff bytes"
        assert path == str(tmp_path / "PR_20101217_025403_165_LC16.tif")

    def test_failed_response_writes_nothing(self, tmp_path, caplog):
        class FakeResponse:
            ok = False

        class FakeSession:
            def get(self, url):
                return FakeResponse()

        prep.image_fetcher.fetch_image(EXPECTED_URL, str(tmp_path), FakeSession())

        assert list(tmp_path.iterdir()) == []
        assert "Failed to fetch image" in caplog.text

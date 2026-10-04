"""Work around AEMO file-naming changes that nemosis 3.7 / nemed 0.3.3 don't handle.

1. MMSDM archive tables (GENUNITS, DUALLOC). From 2024-08 onwards AEMO names
   monthly archive files like
       PUBLIC_ARCHIVE#GENUNITS#FILE01#202506010000.zip
   instead of the old
       PUBLIC_DVD_GENUNITS_202506010000.zip
   nemosis 3.7 only knows the old name, so those downloads 404. When the old
   name fails, fetch the new name and save the CSV under the old name, which
   is where nemosis looks for it. Only FILE01 is fetched; that covers small
   tables like GENUNITS and DUALLOC (all that nemed.get_marginal_emissions
   needs), not large tables split across several files.

2. NEMDE price setter XMLs. Some files inside AEMO's daily zips lack the
   trailing "00" (e.g. NEMPriceSetter_20260310177.xml for 2026-03-10), so
   nempy can't find them and nemed drops the whole day. Rename them after
   extraction.
"""

import io
import re
import zipfile
from pathlib import Path

import requests
from nemed.helper_functions import mod_xml_cache
from nemosis import downloader

_OLD_NAME = re.compile(r"PUBLIC_DVD_(?P<table>.+)_(?P<stamp>\d{12})\.zip$")
_original_download_unzip_csv = downloader.download_unzip_csv

_SHORT_XML_NAME = re.compile(r"(NEMPriceSetter_\d{11})\.xml")
_original_download_xml = mod_xml_cache.modpricesetter_download_xml_from_nemweb


def _download_unzip_csv(url, down_load_to):
    try:
        return _original_download_unzip_csv(url, down_load_to)
    except zipfile.BadZipFile:
        # 404 page instead of a zip: retry with the new file name below
        match = _OLD_NAME.search(url)
        if match is None:
            raise

    table, stamp = match["table"], match["stamp"]
    new_url = url[: match.start()] + f"PUBLIC_ARCHIVE%23{table}%23FILE01%23{stamp}.zip"
    r = requests.get(new_url, headers=downloader.USR_AGENT_HEADER)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        csv_bytes = z.read(z.namelist()[0])
    Path(down_load_to, f"PUBLIC_DVD_{table}_{stamp}.CSV").write_bytes(csv_bytes)


def _download_xml_from_nemweb(self):
    _original_download_xml(self)
    for path in Path(self.cache_folder).glob("NEMPriceSetter_*.xml"):
        match = _SHORT_XML_NAME.fullmatch(path.name)
        if match:
            path.rename(path.with_name(match[1] + "00.xml"))


def apply():
    downloader.download_unzip_csv = _download_unzip_csv
    # nemed looks this name up each time it reconfigures nempy's XMLCacheManager
    mod_xml_cache.modpricesetter_download_xml_from_nemweb = _download_xml_from_nemweb

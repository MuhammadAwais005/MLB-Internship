# Dataset sources and attribution

The eight source camera photos are from
[Daniel Gatis — docscan](https://github.com/danielgatis/docscan), directory
`examples/`, retrieved 10 November 2026. The repository's MIT license is retained
verbatim in `licenses/docscan-MIT.txt`. The source examples are used as practice
data; this project does not import or copy the source application's scanner code.

Original photo names: `doc-1.jpg`, `doc-2.png`, `doc-3.jpg` through `doc-8.jpg`.
Three straight archival scans/digital reproductions are from Wikimedia Commons:

1. [Bill of Rights](https://commons.wikimedia.org/wiki/File:United_States_Bill_of_Rights.jpg): US National Archives, labelled public domain on the source page.
2. [Constitution, page 1](https://commons.wikimedia.org/wiki/File:Constitution_of_the_United_States,_page_1.jpg): US National Archives, labelled public domain on the source page.
3. [Roosevelt's reply to Einstein](https://commons.wikimedia.org/wiki/File:Roosevelt-einstein-letter.png): Franklin D. Roosevelt, 19 October 1939; source describes a scan and labels the federal-government work public domain.

Downloaded source-sized previews are retained in `source_scans/`. Wikimedia
API metadata, including source license labels, is retained in
`licenses/wikimedia-source-metadata.json`. See each manifest entry for the exact
source page, download URL and image title.

Preprocessing applied to three archival inputs and four additional variants:

- `scan_01.png` through `scan_03.png`: resized archival scans/reproductions with
  a 45px dark margin to make the outer source-page canvas observable.
- `shadow_01.png`, `shadow_02.png`: illumination-field variants of photos 1 and 2.
- `blur_03.png`, `blur_08.png`: mild Gaussian blur on photos 3 and 8.

See `manifest.json` for exact URLs, dimensions, hashes and transformation details.
Original source photos are kept unmodified. Processing overlays are modified
copies with a green boundary. The derivatives and overlays retain source credit
and the included MIT notice.

The sample set is not an independent benchmark or evidence of generalization.
It contains 11 source images plus four variants of source camera scenes, including
repeated document types. The archival inputs have added margins; their supplied
page-canvas rectangle is not a manual annotation of irregular physical paper edges.

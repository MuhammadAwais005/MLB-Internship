# Dataset: 16 images

Real images from public sample sets that computer-vision projects publish on GitHub.
Large phone photos were resized to at most 1600 px on the longest side; image 14 was
cropped to the inside of the light panel. Nothing else was changed.

| # | File | Category | What makes it hard | Source |
|---|---|---|---|---|
| 01 | 01_document_paper_on_desk.jpg | Document | Page photographed at an angle on a wooden desk | OpenCV-Document-Scanner (`desk.JPG`) |
| 02 | 02_document_dark_background.jpg | Document | Page on a dark surface, photographed by phone | OpenCV-Document-Scanner (`cell_pic.jpg`) |
| 03 | 03_document_chart_on_black.jpg | Document | Printed chart on a black background, slight tilt | OpenCV-Document-Scanner (`chart.JPG`) |
| 04 | 04_document_notepad_on_table.jpg | Document | Lined notepad on a table, page almost fills the frame | OpenCV-Document-Scanner (`notepad.JPG`) |
| 05 | 05_document_receipt_dark_table.jpg | Document | Small, crumpled receipt on a dark table | OpenCV-Document-Scanner (`receipt.jpg`) |
| 06 | 06_document_receipt_shadow.jpg | Document + shadow | Receipt fills the frame, shadow across the bottom | PaddleOCR samples (`doc/imgs_en/254.jpg`), also used on Day 29 |
| 07 | 07_document_dollar_bill.jpg | Document | Bill with a low-contrast border on wood | OpenCV-Document-Scanner (`dollar_bill.JPG`) |
| 08 | 08_document_sudoku_uneven_light.png | Document + uneven light | Newspaper puzzle, strong light gradient | OpenCV samples (`sudoku.png`) |
| 09 | 09_object_coins_dark_background.png | Objects, plain background | 24 coins, top of the background is brighter | scikit-image (`data.coins`) |
| 10 | 10_object_candies_white_background.png | Objects, plain background | Many colourful candies, some cut off at the edges | OpenCV samples (`smarties.png`) |
| 11 | 11_object_cookie_box.png | Object, plain background | Grey box on white, object fills most of the frame | OpenCV samples (`box.png`) |
| 12 | 12_object_cartoon_fish.jpg | Object, plain background | Thin fins and bubbles, white parts inside the fish | OpenCV samples (`HappyFish.jpg`) |
| 13 | 13_object_playing_cards.png | Objects, plain background | White cards on a white background | OpenCV samples (`cards.png`) |
| 14 | 14_uneven_light_lightbox_objects.jpg | Uneven light | Dark parts on a light box that is brighter in the middle | OpenCV samples (`pca_test1.jpg`, cropped) |
| 15 | 15_uneven_light_objects_with_shadows.jpg | Uneven light + shadows | Small objects on a table, bright left corner, soft shadows | OpenCV samples (`stuff.jpg`) |
| 16 | 16_shadow_blocks_on_table.jpg | Shadows | Grey blocks casting shadows, two-tone background | OpenCV samples (`blox.jpg`) |

Repositories:
- OpenCV samples: https://github.com/opencv/opencv/tree/4.x/samples/data (Apache 2.0)
- OpenCV-Document-Scanner: https://github.com/andrewdcampbell/OpenCV-Document-Scanner (`sample_images/`)
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR (Apache 2.0)
- scikit-image test data: https://scikit-image.org (BSD)

Used for learning and testing only. Check each source before reusing the images publicly.

**Add your own:** put any `.jpg`, `.png`, `.bmp`, `.tif` or `.webp` file in this folder
and run `python main.py --batch dataset`, or upload it in the web app.

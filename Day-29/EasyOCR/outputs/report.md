# EasyOCR Results Report

- Images tested: 12
- Methods compared: original, grayscale, enhanced, binarized
- Device: CPU

## Average per method

| method    |   avg_word_accuracy |   avg_char_accuracy |   avg_confidence |   avg_seconds |
|:----------|--------------------:|--------------------:|-----------------:|--------------:|
| original  |                92.8 |                96.5 |             86.1 |          10.2 |
| grayscale |                92.8 |                97   |             85.5 |          10.1 |
| enhanced  |                95   |                97.1 |             87.3 |          11.2 |
| binarized |                90.6 |                95.7 |             83.9 |          10.9 |

## Word accuracy (%) per image

Share of the correct words that EasyOCR found. Main metric, because it
does not depend on the order in which boxes come back.

| image                          |   original |   grayscale |   enhanced |   binarized |
|:-------------------------------|-----------:|------------:|-----------:|------------:|
| 01_printed_document.jpg        |      100   |       100   |      100   |        98.4 |
| 02_printed_document_shadow.jpg |       91.5 |        87.2 |      100   |        97.9 |
| 03_receipt.jpg                 |       96.5 |        96.5 |       98.2 |        96.5 |
| 04_invoice.jpg                 |       98   |        98   |       98   |       100   |
| 05_signboard_street.jpg        |       81.2 |        81.2 |       81.2 |        87.5 |
| 06_poster_notice.png           |      100   |       100   |      100   |       100   |
| 07_book_page.jpg               |      100   |       100   |      100   |        95.8 |
| 08_handwritten_note.jpg        |       82.6 |        87   |       91.3 |        65.2 |
| 09_handwritten_list.jpg        |       86.7 |        86.7 |       86.7 |        80   |
| 10_shop_signboard.jpg          |      100   |       100   |      100   |       100   |
| 11_blurry_label.jpg            |       76.9 |        76.9 |       84.6 |        69.2 |
| 12_faded_photocopy.jpg         |      100   |       100   |      100   |        96.4 |

## Character accuracy (%) per image

100 minus Character Error Rate. Stricter: it also catches single wrong
letters, but it drops when lines are rebuilt in a different order
(e.g. the tilted receipt), so read it together with word accuracy.

| image                          |   original |   grayscale |   enhanced |   binarized |
|:-------------------------------|-----------:|------------:|-----------:|------------:|
| 01_printed_document.jpg        |      100   |       100   |      100   |        99.7 |
| 02_printed_document_shadow.jpg |       98.8 |        98.5 |      100   |        99.6 |
| 03_receipt.jpg                 |       73   |        78.8 |       79.2 |        76.1 |
| 04_invoice.jpg                 |       99.3 |        99.3 |       99.3 |       100   |
| 05_signboard_street.jpg        |       95.6 |        94.4 |       92.2 |        92.2 |
| 06_poster_notice.png           |      100   |       100   |      100   |       100   |
| 07_book_page.jpg               |      100   |       100   |      100   |        99.5 |
| 08_handwritten_note.jpg        |       97.4 |        98.3 |       98.3 |        92.2 |
| 09_handwritten_list.jpg        |       98.8 |        98.8 |       98.8 |        97.5 |
| 10_shop_signboard.jpg          |      100   |       100   |      100   |       100   |
| 11_blurry_label.jpg            |       95.5 |        95.5 |       97   |        92.4 |
| 12_faded_photocopy.jpg         |      100   |       100   |      100   |        99.5 |

## Best method per image (word accuracy)

| Image | Best method | Original | Best | Gain |
|---|---|---|---|---|
| 01_printed_document.jpg | original, grayscale, enhanced | 100.0 | 100.0 | +0.0 |
| 02_printed_document_shadow.jpg | enhanced | 91.5 | 100.0 | +8.5 |
| 03_receipt.jpg | enhanced | 96.5 | 98.2 | +1.7 |
| 04_invoice.jpg | binarized | 98.0 | 100.0 | +2.0 |
| 05_signboard_street.jpg | binarized | 81.2 | 87.5 | +6.3 |
| 06_poster_notice.png | original, grayscale, enhanced, binarized | 100.0 | 100.0 | +0.0 |
| 07_book_page.jpg | original, grayscale, enhanced | 100.0 | 100.0 | +0.0 |
| 08_handwritten_note.jpg | enhanced | 82.6 | 91.3 | +8.7 |
| 09_handwritten_list.jpg | original, grayscale, enhanced | 86.7 | 86.7 | +0.0 |
| 10_shop_signboard.jpg | original, grayscale, enhanced, binarized | 100.0 | 100.0 | +0.0 |
| 11_blurry_label.jpg | enhanced | 76.9 | 84.6 | +7.7 |
| 12_faded_photocopy.jpg | original, grayscale, enhanced | 100.0 | 100.0 | +0.0 |

Full extracted text for every image and method is in `outputs/text/`.

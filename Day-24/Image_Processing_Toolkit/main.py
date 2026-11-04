
import os
from image_toolkit import ImageToolkit


# Initialize application
toolkit = ImageToolkit()

# Create output folder
os.makedirs("output", exist_ok=True)


def display_menu():

    print("\n" + "=" * 45)
    print("       IMAGE PROCESSING TOOLKIT")
    print("=" * 45)

    print("1.  Load Image")
    print("2.  Convert to Grayscale")
    print("3.  Resize Image")
    print("4.  Rotate Image")
    print("5.  Flip Image")
    print("6.  Crop Image")
    print("7.  Draw Shapes")
    print("8.  Add Custom Text")
    print("9.  Adjust Brightness")
    print("10. Adjust Contrast")
    print("11. Compare BGR and RGB")
    print("12. Compare Original and Processed")
    print("13. Save Processed Image")
    print("0.  Exit")

    print("=" * 45)


while True:

    display_menu()

    choice = input("Enter your choice: ").strip()

    try:

        if choice == "1":

            path = input("Enter image path: ").strip().strip('"')
            toolkit.load_image(path)

        elif choice == "2":

            toolkit.grayscale()

        elif choice == "3":

            width = int(input("Enter new width: "))
            height = int(input("Enter new height: "))

            toolkit.resize(width, height)

        elif choice == "4":

            angle = int(input("Enter angle (90, 180, 270): "))

            if angle not in [90, 180, 270]:
                print("Invalid angle!")
            else:
                toolkit.rotate(angle)

        elif choice == "5":

            print("1. Horizontal")
            print("2. Vertical")
            print("3. Both")

            flip_choice = input("Choose flip direction: ")

            directions = {
                "1": "horizontal",
                "2": "vertical",
                "3": "both"
            }

            if flip_choice in directions:
                toolkit.flip(directions[flip_choice])
            else:
                print("Invalid flip choice!")

        elif choice == "6":

            print("Enter crop coordinates:")

            x1 = int(input("X1: "))
            y1 = int(input("Y1: "))
            x2 = int(input("X2: "))
            y2 = int(input("Y2: "))

            toolkit.crop(x1, y1, x2, y2)

        elif choice == "7":

            toolkit.draw_shapes()

        elif choice == "8":

            text = input("Enter your custom text: ")
            toolkit.add_text(text)

        elif choice == "9":

            value = int(input("Brightness adjustment (-100 to 100): "))

            if -100 <= value <= 100:
                toolkit.adjust_brightness(value)
            else:
                print("Choose a value between -100 and 100.")

        elif choice == "10":

            value = float(input("Contrast factor (0.1 to 3.0): "))

            if 0.1 <= value <= 3.0:
                toolkit.adjust_contrast(value)
            else:
                print("Choose a value between 0.1 and 3.0.")

        elif choice == "11":

            toolkit.compare_color_channels()

        elif choice == "12":

            toolkit.compare_images()

        elif choice == "13":

            filename = input(
                "Enter output filename (e.g. processed.jpg): "
            ).strip()

            if not filename:
                filename = "processed.jpg"

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            ):
                filename += ".jpg"

            toolkit.save_image(os.path.join("output", filename))

        elif choice == "0":

            print("Thank you for using Image Processing Toolkit!")
            break

        else:

            print("Invalid choice! Please try again.")

    except ValueError as error:
        print("Input/operation error:", error)

    except Exception as error:
        print("Something went wrong:", error)

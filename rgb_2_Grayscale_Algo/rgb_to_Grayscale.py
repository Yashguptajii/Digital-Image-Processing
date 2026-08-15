from PIL import Image
import os

def average_method(r, g, b):
    return int((r + g + b) / 3)

def lightness_method(r, g, b):
    return int((max(r, g, b) + min(r, g, b)) / 2)

def luminosity_method(r, g, b):
    return int(0.21 * r + 0.72 * g + 0.07 * b)

def bt601_method(r, g, b):
    return int(0.299 * r + 0.587 * g + 0.114 * b)

def convert_image(image, algorithm):
    width, height = image.size
    output = Image.new("L", (width, height))

    input_pixels = image.load()
    output_pixels = output.load()

    for y in range(height):
        for x in range(width):
            r, g, b = input_pixels[x, y]
            output_pixels[x, y] = algorithm(r, g, b)

    return output

input_file = "input.jpg"
output_folder = "grayscale_outputs"

os.makedirs(output_folder, exist_ok=True)

image = Image.open(input_file).convert("RGB")

average = convert_image(image, average_method)
lightness = convert_image(image, lightness_method)
luminosity = convert_image(image, luminosity_method)
bt601 = convert_image(image, bt601_method)

average.save(os.path.join(output_folder, "average.png"))
lightness.save(os.path.join(output_folder, "lightness.png"))
luminosity.save(os.path.join(output_folder, "luminosity.png"))
bt601.save(os.path.join(output_folder, "bt601.png"))

print("Grayscale conversion completed.")
print("Output files saved in:", output_folder)

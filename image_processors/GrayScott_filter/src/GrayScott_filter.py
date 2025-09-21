from tkinter import Tk, Label, Button, Entry, filedialog, Canvas, PhotoImage
from PIL import Image, ImageTk, ImageFilter
import utils

class GrayScottFilterApp:
    def __init__(self, master):
        self.master = master
        master.title("Gray-Scott Filter")

        self.label = Label(master, text="Load Image for Gray-Scott Processing:")
        self.label.pack()

        self.load_button = Button(master, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.iterations_label = Label(master, text="Number of iterations:")
        self.iterations_label.pack()

        self.iterations_entry = Entry(master)
        self.iterations_entry.pack()

        self.process_button = Button(master, text="Process Image", command=self.process_image)
        self.process_button.pack()

        self.save_button = Button(master, text="Save Image", command=self.save_image)
        self.save_button.pack()

        self.canvas = Canvas(master)
        self.canvas.pack()

        self.image = None
        self.processed_image = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path)
            # Resize image if it's larger than 1024 pixels in either dimension
            if self.image.width > 1024 or self.image.height > 1024:
                self.image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            self.display_image(self.image)

    def display_image(self, image):
        self.processed_image = image
        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)
        self.canvas.config(width=image.width, height=image.height)

    def process_image(self):
        if self.image:
            try:
                iterations = int(self.iterations_entry.get())
                # Convert to greyscale as first step
                processed = self.image.convert('L')
                for _ in range(iterations):
                    processed = utils.sharpen_image(processed)
                    processed = utils.sharpen_image(processed) # applying sharpening multiple times
                    processed = utils.blur_image(processed)
                self.display_image(processed)
            except ValueError:
                print("Please enter a valid number of iterations.")

    def save_image(self):
        if self.processed_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                       filetypes=[("PNG files", "*.png"),
                                                                  ("JPEG files", "*.jpg"),
                                                                  ("All files", "*.*")])
            if file_path:
                self.processed_image.save(file_path)

if __name__ == "__main__":
    root = Tk()
    app = GrayScottFilterApp(root)
    root.mainloop()

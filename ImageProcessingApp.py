from tkinter import *
from tkinter import filedialog, messagebox, font

def linear_mapping(image, start_val, end_val, initial_val, slope):

    transformed_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            original_pixel_value = image[i, j]
            if start_val <= original_pixel_value  and original_pixel_value <= end_val:
                new_pixel_value = int((original_pixel_value-start_val)*slope + initial_val)
                if new_pixel_value > 255 and new_pixel_value < 0:
                    new_pixel_value = 0
                transformed_image[i, j] = new_pixel_value

    return transformed_image.astype(np.uint8)

def nonlinear_mapping(image, segments):
    transformed_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            original_pixel_value = image[i, j]
            for seg in segments:
                start_val, end_val, initial_val, slope = seg
                if start_val <= original_pixel_value and original_pixel_value <= end_val:
                    new_pixel_value = int((original_pixel_value-start_val)*slope + initial_val)
                    if new_pixel_value > 255 and new_pixel_value < 0:
                        new_pixel_value = 0
                    transformed_image[i, j] = new_pixel_value
                    break
            else:
                transformed_image[i, j] = original_pixel_value

    return transformed_image.astype(np.uint8)

def histogram_stretch(image):
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    stretched = np.uint8((image - min_intensity) * 255.0 / (max_intensity - min_intensity))
    return stretched

def histogram_compress(image, compressed_min, compressed_max):
    stretched = histogram_stretch(image)
    compressed = np.uint8((compressed_max - compressed_min) / 255 * (stretched - np.min(stretched)) + compressed_min)
    return compressed

def digital_negative(image):
    transformed_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_value = image[i, j]
            transformed_image[i, j] = 255 - pixel_value
    return transformed_image


import cv2

def ACE_Filter(image, n, k1, k2, clip_lower=0, clip_upper=255):
    try:
        mean_image = np.mean(image)
        padded_image = np.pad(image, ((n // 2, n // 2), (n // 2, n // 2)), mode='constant')

        transformed_image = np.zeros_like(image, dtype=np.float32)
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                window = padded_image[r:r + n, c:c + n]
                local_mean = np.mean(window)
                local_std = np.sqrt(np.sum((window - local_mean) ** 2) / (n ** 2 - 1))
                ace_value = k1 * (mean_image / local_std) * (image[r, c] - local_mean) + k2 * local_mean
                ace_value = np.clip(ace_value, clip_lower, clip_upper)
                transformed_image[r, c] = ace_value
        transformed_image = np.uint8(transformed_image)

        return transformed_image
    except Exception as e:
        print("Error in ACE_Filter:", e)
        return None


def histogram_specification(image, desired_histogram):
    # Compute the histograms of the original image and the desired histogram
    hist_original, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    hist_desired, _ = np.histogram(desired_histogram.flatten(), bins=256, range=[0, 256])

    # Compute cumulative distribution functions (CDFs) for both histograms
    cdf_original = hist_original.cumsum()
    cdf_desired = hist_desired.cumsum()

    # Normalize the CDFs to be in the range [0, 255]
    cdf_original = (cdf_original - cdf_original.min()) * 255 / (cdf_original.max() - cdf_original.min())
    cdf_desired = (cdf_desired - cdf_desired.min()) * 255 / (cdf_desired.max() - cdf_desired.min())

    # Initialize an array to store the mapping from original histogram to desired histogram
    mapping = np.zeros(256, dtype=np.uint8)

    # Map each intensity value in the original histogram to the closest intensity value in the desired histogram
    for i in range(256):
        mapping[i] = np.argmin(np.abs(cdf_desired - cdf_original[i]))

    # Apply the mapping to the original image
    specified_histogram_image = mapping[image]

    return specified_histogram_image


import numpy as np

def pseudomedian(sequence):
    sorted_sequence = np.sort(sequence)
    M = len(sequence) // 2
    MAXIMIN = max(sorted_sequence[:M+1]) - min(sorted_sequence[:M+1])
    MINIMAX = max(sorted_sequence[M:]) - min(sorted_sequence[M:])
    PMED = 0.5 * (MINIMAX) + 0.5 * MAXIMIN
    return PMED


def pseudomedian_filter(image, window_size):
    padded_image = np.pad(image, pad_width=window_size // 2, mode='reflect')
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded_image[i:i + window_size, j:j + window_size].flatten()
            median_value = np.median(neighborhood)  # Compute the median instead of pseudomedian
            filtered_image[i, j] = median_value

    return filtered_image.astype(np.uint8)



def add_gaussian_noise(image, noise_variance):
    mean = 0
    std_dev = np.sqrt(noise_variance)
    noise = np.random.normal(mean, std_dev, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def mmse_filter(image, noise_variance, kernel_size):
    # Calculate local variance
    local_variance = cv2.GaussianBlur(image.astype(np.float32) ** 2, (kernel_size, kernel_size), 0) - cv2.GaussianBlur(
        image.astype(np.float32), (kernel_size, kernel_size), 0) ** 2

    weights = noise_variance / (local_variance + noise_variance)

    filtered_image = (1 - weights) * image + weights * cv2.blur(image, (kernel_size, kernel_size))

    return filtered_image.astype(np.uint8)


def linear_mapping_gui():
    linear_mapping_window = Toplevel(root)
    linear_mapping_window.title("Linear Mapping")
    linear_mapping_window.config(bg="#2E3440")  # Set background color to Dark blue-gray

    # Define font object with increased size
    button_font = font.Font(size=14)

    # Label for parameter input with Light gray text on Dark blue-gray background
    label = Label(linear_mapping_window, text="Enter the parameters for linear mapping:", font=button_font,
                  bg="#2E3440", fg="#ECEFF4")
    label.pack()

    # Label and entry for start value with Light gray text on Dark blue-gray background
    start_val_label = Label(linear_mapping_window, text="Start Value:", font=button_font, bg="#2E3440", fg="#ECEFF4")
    start_val_label.pack()
    start_val_entry = Entry(linear_mapping_window)
    start_val_entry.pack()

    # Label and entry for end value with Light gray text on Dark blue-gray background
    end_val_label = Label(linear_mapping_window, text="End Value:", font=button_font, bg="#2E3440", fg="#ECEFF4")
    end_val_label.pack()
    end_val_entry = Entry(linear_mapping_window)
    end_val_entry.pack()

    # Label and entry for initial value with Light gray text on Dark blue-gray background
    initial_val_label = Label(linear_mapping_window, text="Initial Value:", font=button_font, bg="#2E3440", fg="#ECEFF4")
    initial_val_label.pack()
    initial_val_entry = Entry(linear_mapping_window)
    initial_val_entry.pack()

    # Label and entry for slope with Light gray text on Dark blue-gray background
    slope_label = Label(linear_mapping_window, text="Slope:", font=button_font, bg="#2E3440", fg="#ECEFF4")
    slope_label.pack()
    slope_entry = Entry(linear_mapping_window)
    slope_entry.pack()

    def apply_linear_mapping():
        start_val = float(start_val_entry.get())
        end_val = float(end_val_entry.get())
        initial_val = float(initial_val_entry.get())
        slope = float(slope_entry.get())
        transformed_image = linear_mapping(image_np, start_val, end_val, initial_val, slope)
        cv2.imshow("Original Image", image_np)
        cv2.imshow("Transformed Image", transformed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Apply button with Light blue-gray color
    apply_button = Button(linear_mapping_window, text="Apply", command=apply_linear_mapping, font=button_font,
                          bg="#4C566A", fg="#ECEFF4")  # Use Light gray text on Light blue-gray background
    apply_button.pack()
def nonlinear_mapping_gui():
    nonlinear_mapping_window = Toplevel(root)
    nonlinear_mapping_window.title("Nonlinear Mapping")
    nonlinear_mapping_window.config(bg="#2E3440")

    button_font = font.Font(size=14)

    label = Label(nonlinear_mapping_window, text="Enter the parameters for nonlinear mapping:", font=button_font,
                  bg="#2E3440", fg="#ECEFF4")
    label.pack()

    def add_segment():
        segment_frame = Frame(nonlinear_mapping_window, bg="#2E3440")
        segment_frame.pack()

        start_val_label = Label(segment_frame, text="Start Value:", font=button_font, bg="#2E3440", fg="#ECEFF4")
        start_val_label.pack(side=LEFT)
        start_val_entry = Entry(segment_frame)
        start_val_entry.pack(side=LEFT)

        end_val_label = Label(segment_frame, text="End Value:", font=button_font, bg="#2E3440", fg="#ECEFF4")
        end_val_label.pack(side=LEFT)
        end_val_entry = Entry(segment_frame)
        end_val_entry.pack(side=LEFT)

        initial_val_label = Label(segment_frame, text="Initial Value:", font=button_font, bg="#2E3440", fg="#ECEFF4")
        initial_val_label.pack(side=LEFT)
        initial_val_entry = Entry(segment_frame)
        initial_val_entry.pack(side=LEFT)

        slope_label = Label(segment_frame, text="Slope:", font=button_font, bg="#2E3440", fg="#ECEFF4")
        slope_label.pack(side=LEFT)
        slope_entry = Entry(segment_frame)
        slope_entry.pack(side=LEFT)

        segment_entries.append((start_val_entry, end_val_entry, initial_val_entry, slope_entry))

    segment_entries = []
    add_segment()

    add_button = Button(nonlinear_mapping_window, text="Add Segment", command=add_segment, font=button_font,
                        bg="#4C566A", fg="#ECEFF4")
    add_button.pack()

    def apply_nonlinear_mapping():
        segments = []
        for entry_set in segment_entries:
            start_val = float(entry_set[0].get())
            end_val = float(entry_set[1].get())
            initial_val = float(entry_set[2].get())
            slope = float(entry_set[3].get())
            segments.append((start_val, end_val, initial_val, slope))

        transformed_image = nonlinear_mapping(image_np, segments)
        cv2.imshow("Original Image", image_np)
        cv2.imshow("Transformed Image", transformed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    apply_button = Button(nonlinear_mapping_window, text="Apply", command=apply_nonlinear_mapping, font=button_font,
                          bg="#4C566A", fg="#ECEFF4")
    apply_button.pack()

def histogram_compress_gui():
    histogram_compress_window = Toplevel(root)
    histogram_compress_window.title("Histogram Compression")
    histogram_compress_window.config(bg="#2E3440")

    button_font = font.Font(size=14)


    label = Label(histogram_compress_window, text="Enter the parameters for histogram compression:", font=button_font,
                  bg="#2E3440", fg="#ECEFF4")
    label.pack()

    compressed_min_label = Label(histogram_compress_window, text="Compressed Min Value:", font=button_font,
                                 bg="#2E3440", fg="#ECEFF4")
    compressed_min_label.pack()
    compressed_min_entry = Entry(histogram_compress_window)
    compressed_min_entry.pack()

    compressed_max_label = Label(histogram_compress_window, text="Compressed Max Value:", font=button_font,
                                 bg="#2E3440", fg="#ECEFF4")
    compressed_max_label.pack()
    compressed_max_entry = Entry(histogram_compress_window)
    compressed_max_entry.pack()


    def apply_histogram_compress():
        compressed_min = int(compressed_min_entry.get())
        compressed_max = int(compressed_max_entry.get())
        transformed_image = histogram_compress(image_np, compressed_min, compressed_max)
        cv2.imshow("Original Image", image_np)
        cv2.imshow("Transformed Image", transformed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    apply_button = Button(histogram_compress_window, text="Apply", command=apply_histogram_compress, font=button_font,
                          bg="#4C566A", fg="#ECEFF4")
    apply_button.pack()
def histogram_stretch_gui():
    transformed_image = histogram_stretch(image_np)
    cv2.imshow("Original Image",image_np)
    cv2.imshow("Transformed Image", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def digital_negative_gui():
    transformed_image = digital_negative(image_np)
    cv2.imshow("Original Image", image_np)
    cv2.imshow("Transformed Image", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ACE_Filter_gui():
    ACE_Filter_window = Toplevel(root)
    ACE_Filter_window.title("ACE Filter")
    ACE_Filter_window.config(bg="#2E3440")
    button_font = font.Font(size=14)


    label = Label(ACE_Filter_window, text="Enter the parameters for ACE Filter:", font=button_font,
                  bg="#2E3440", fg="#ECEFF4")
    label.pack()

    window_size_label = Label(ACE_Filter_window, text="Window size:", font=button_font,
                              bg="#2E3440", fg="#ECEFF4")
    window_size_label.pack()
    window_size_entry = Entry(ACE_Filter_window)
    window_size_entry.pack()


    k1_label = Label(ACE_Filter_window, text="k1:", font=button_font,
                     bg="#2E3440", fg="#ECEFF4")
    k1_label.pack()
    k1_entry = Entry(ACE_Filter_window)
    k1_entry.pack()

    # Label and entry for k2
    k2_label = Label(ACE_Filter_window, text="k2:", font=button_font,
                     bg="#2E3440", fg="#ECEFF4")
    k2_label.pack()
    k2_entry = Entry(ACE_Filter_window)
    k2_entry.pack()


    def apply_ACE_FILTER():
        window_size_val = int(window_size_entry.get())
        k1_val = float(k1_entry.get())
        k2_val = float(k2_entry.get())
        transformed_image = ACE_Filter(image_np, window_size_val, k1_val, k2_val)
        if transformed_image is not None:
            cv2.imshow("Original Image", image_np)
            cv2.imshow("Transformed Image", transformed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    apply_button = Button(ACE_Filter_window, text="Apply", command=apply_ACE_FILTER, font=button_font,
                          bg="#4C566A", fg="#ECEFF4")
    apply_button.pack()
def histogram_specification_gui():
    def apply_histogram_specification():
        if not image_path:
            messagebox.showerror("Error", "No original image selected.")
            return

        desired_histogram_path = filedialog.askopenfilename()
        if desired_histogram_path:
            desired_histogram_image = cv2.imread(desired_histogram_path, cv2.IMREAD_GRAYSCALE)
            if desired_histogram_image is None:
                messagebox.showerror("Error", "Failed to load the desired histogram image.")
                return
        else:
            messagebox.showerror("Error", "No desired histogram image selected.")
            return


        if desired_histogram_image.shape != image_np.shape:

            desired_histogram_image = cv2.resize(desired_histogram_image, (image_np.shape[1], image_np.shape[0]))

        transformed_image = histogram_specification(image_np, desired_histogram_image)
        if transformed_image is not None:
            cv2.imshow("Original Image", image_np)
            cv2.imshow("Target Image: ", desired_histogram_image)
            cv2.imshow("Transformed Image", transformed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    histogram_specification_window = Toplevel(root)
    histogram_specification_window.title("Histogram Specification")
    histogram_specification_window.config(bg="#2E3440")


    button_font = font.Font(size=14)

    apply_button = Button(histogram_specification_window, text="Select target image", command=apply_histogram_specification, font=button_font,
                          bg="#4C566A", fg="#ECEFF4")
    apply_button.pack()
def pseudomedian_filter_gui():
    pseudomedian_filter_window = Toplevel(root)
    pseudomedian_filter_window.title("Pseudomedian Filter")
    pseudomedian_filter_window.config(bg="#2E3440")
    button_font = font.Font(size=14)


    label = Label(pseudomedian_filter_window, text="Enter the window size for pseudomedian filter:", font=button_font,
                  bg="#2E3440", fg="#ECEFF4")
    label.pack()

    window_size_entry = Entry(pseudomedian_filter_window)
    window_size_entry.pack()

    def apply_pseudomedian_filter():
        try:
            window_size = int(window_size_entry.get())
            filtered_image = pseudomedian_filter(image_np, window_size)
            cv2.imshow("Original Image", image_np)
            cv2.imshow("Filtered Image (Pseudomedian)", filtered_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


    apply_button = Button(pseudomedian_filter_window, text="Apply", command=apply_pseudomedian_filter, font=button_font,
                          bg="#4C566A", fg="#ECEFF4")
    apply_button.pack()


def mmse_filter_gui():
    mmse_filter_window = Toplevel(root)
    mmse_filter_window.title("MMSE Filter")
    mmse_filter_window.config(bg="#2E3440")
    button_font = font.Font(size=14)


    label = Label(mmse_filter_window, text="Enter the noise variance for MMSE filter:", font=button_font,
                  bg="#2E3440", fg="#ECEFF4")
    label.pack()

    noise_variance_entry = Entry(mmse_filter_window)
    noise_variance_entry.pack()


    label = Label(mmse_filter_window, text="Enter the kernel size:", font=button_font,
                  bg="#2E3440", fg="#ECEFF4")
    label.pack()

    kernel_entry = Entry(mmse_filter_window)
    kernel_entry.pack()

    def apply_mmse_filter():
        try:
            noise_variance = float(noise_variance_entry.get())
            kernel_size = int(kernel_entry.get())

            noisy_image = add_gaussian_noise(image_np, noise_variance)

            filtered_image = mmse_filter(noisy_image, noise_variance, kernel_size)

            cv2.imshow("Original Image", image_np)
            cv2.imshow("Noisy Image (Gaussian)", noisy_image)
            cv2.imshow("Filtered Image (MMSE)", filtered_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    apply_button = Button(mmse_filter_window, text="Apply", command=apply_mmse_filter, font=button_font,
                          bg="#4C566A", fg="#ECEFF4")
    apply_button.pack()

def select_image():
    global image_np
    global image_path
    image_path = filedialog.askopenfilename()
    if image_path:
        image_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image_np is None:
            messagebox.showerror("Error", "Failed to load the image.")
        else:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        messagebox.showerror("Error", "No image selected.")

root = Tk()
root.title("Image Processing App")
root.geometry("800x600")


# Define colors
bg_color = "#2E3440"  # Dark blue-gray
button_bg_color = "#4C566A"  # Light blue-gray
button_fg_color = "#ECEFF4"  # Light gray
button_hover_bg_color = "#5E81AC"  # Dark blue

# Frame for image selection
select_frame = Frame(root, bg=bg_color)
select_frame.pack(fill="x", padx=10, pady=10)

select_image_button = Button(select_frame, text="Select Image", command=select_image, font=("Helvetica", 14, "bold"), bg=button_bg_color, fg=button_fg_color, activebackground=button_hover_bg_color)
select_image_button.pack(side="left", padx=5)

# Frame for image processing options
options_frame = Frame(root, bg=bg_color)
options_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

# Buttons for image processing options
button_font = font.Font(size=12, weight="bold")  # Define font object with increased size and bold weight

linear_mapping_button = Button(options_frame, text="Linear Mapping", command=linear_mapping_gui, font=button_font, bg=button_bg_color, fg=button_fg_color, activebackground=button_hover_bg_color)
linear_mapping_button.pack(pady=5, padx=10, ipadx=10, ipady=5)

nonlinear_mapping_button = Button(options_frame, text="Nonlinear Mapping", command=nonlinear_mapping_gui, font=button_font, bg=button_bg_color, fg=button_fg_color, activebackground=button_hover_bg_color)
nonlinear_mapping_button.pack(pady=5, padx=10, ipadx=10, ipady=5)

histogram_compress_button = Button(options_frame, text="Histogram Compression", command=histogram_compress_gui, font=button_font, bg=button_bg_color, fg=button_fg_color, activebackground=button_hover_bg_color)
histogram_compress_button.pack(pady=5, padx=10, ipadx=10, ipady=5)

histogram_stretch_button = Button(options_frame, text="Histogram Stretching", command=histogram_stretch_gui, font=button_font, bg=button_bg_color, fg=button_fg_color, activebackground=button_hover_bg_color)
histogram_stretch_button.pack(pady=5, padx=10, ipadx=10, ipady=5)

digital_negative_button = Button(options_frame, text="Digital Negative", command=digital_negative_gui, font=button_font, bg=button_bg_color, fg=button_fg_color, activebackground=button_hover_bg_color)
digital_negative_button.pack(pady=5, padx=10, ipadx=10, ipady=5)

ACE_Filter_button = Button(options_frame, text="ACE Filter", command=ACE_Filter_gui, font=button_font, bg=button_bg_color, fg=button_fg_color, activebackground=button_hover_bg_color)
ACE_Filter_button.pack(pady=5, padx=10, ipadx=10, ipady=5)

histogram_specification_button = Button(options_frame, text="Histogram Specification", command=histogram_specification_gui, font=button_font, bg=button_bg_color, fg=button_fg_color, activebackground=button_hover_bg_color)
histogram_specification_button.pack(pady=5, padx=10, ipadx=10, ipady=5)

pseudomedian_filter_button = Button(options_frame, text="Pseudomedian Filter", command=pseudomedian_filter_gui, font=button_font, bg=button_bg_color, fg=button_fg_color, activebackground=button_hover_bg_color)
pseudomedian_filter_button.pack(pady=5, padx=10, ipadx=10, ipady=5)

mmse_filter_button = Button(options_frame, text="MMSE Filter", command=mmse_filter_gui, font=button_font, bg=button_bg_color, fg=button_fg_color, activebackground=button_hover_bg_color)
mmse_filter_button.pack(pady=5, padx=10, ipadx=10, ipady=5)

root.mainloop()
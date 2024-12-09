import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from PIL import Image, ImageTk

# Funções utilitárias
def load_image():
    global img
    file_path = filedialog.askopenfilename()
    if file_path:
        img = np.array(Image.open(file_path).convert("RGB"))
        display_image(img, original=True)
        refresh_canvas()

def display_image(img, original=False):
    img_resized = resize_image(img, 500, 500)
    img_tk = ImageTk.PhotoImage(Image.fromarray(img_resized))
    canvas = original_image_canvas if original else edited_image_canvas
    canvas.delete("all")
    canvas.img_ref = img_tk
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

def resize_image(image, max_width, max_height):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return np.array(Image.fromarray(image).resize((new_w, new_h)))

def apply_filter(filter_type):
    global img
    if img is None:
        print("Nenhuma imagem carregada.")
        return

    # Mostrar "Processando..."
    loading_label.config(text="Processando...")
    root.update_idletasks()  # Atualiza a interface imediatamente

    # Inicializar variável para a imagem filtrada
    filtered = img

    # Processamento do filtro
    if filter_type == "Low Pass":
        kernel_size = radius_scale.get() * 2 + 1
        kernel = create_gaussian_kernel(kernel_size, kernel_size / 5)
        filtered = manual_convolution(img, kernel)
    elif filter_type == "High Pass":
        filtered = high_pass_filter(img, high_pass_var.get())
    elif filter_type == "Mean":
        kernel_size = radius_scale.get()
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
        filtered = manual_convolution(img, kernel)
    elif filter_type == "Segmentation":
        threshold = threshold_scale.get()
        filtered = segmentation_filter(img, threshold)
    elif filter_type == "Morphology":
        morph_type = morph_type_var.get()
        kernel_type = kernel_type_var.get()
        print(f"Aplicando Morfologia: {morph_type} com Kernel: {kernel_type}")
        filtered = morphology_filter(img, morph_type, kernel_type)
    elif filter_type == "Binary Threshold":
        filtered = binary_threshold(img, threshold_scale.get())
    elif filter_type == "Adaptive Threshold":
        filtered = adaptive_threshold(img)
    else:
        print(f"Filtro {filter_type} não é reconhecido.")
        loading_label.config(text="")
        return

    # Atualizar a imagem no canvas e esconder "Processando..."
    display_image(filtered, original=False)
    loading_label.config(text="")

def create_gaussian_kernel(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    return kernel / kernel.sum()

def manual_convolution(image, kernel):
    pad = kernel.shape[0] // 2
    if len(image.shape) == 2:  # Imagem em escala de cinza
        h, w = image.shape
        output = np.zeros_like(image, dtype=np.float32)
        padded = np.pad(image, pad, mode="constant")
        for y in range(h):
            for x in range(w):
                region = padded[y:y + kernel.shape[0], x:x + kernel.shape[1]]
                output[y, x] = np.sum(region * kernel)
    elif len(image.shape) == 3:  # Imagem RGB
        h, w = image.shape[:2]
        output = np.zeros_like(image, dtype=np.float32)
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
        for y in range(h):
            for x in range(w):
                for c in range(3):
                    region = padded[y:y + kernel.shape[0], x:x + kernel.shape[1], c]
                    output[y, x, c] = np.sum(region * kernel)
    else:
        raise ValueError("Imagem de entrada deve ser 2D (escala de cinza) ou 3D (RGB).")

    return np.clip(output, 0, 255).astype(np.uint8)

def high_pass_filter(image, method):
    gray = convert_to_gray(image)  # Converte para escala de cinza
    if method == "Sobel":
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        gx = manual_convolution(gray, kernel_x)
        gy = manual_convolution(gray, kernel_y)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        return np.stack([gradient_magnitude]*3, axis=-1).astype(np.uint8)  # Converte de volta para RGB
    elif method == "Laplacian":
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        laplacian = manual_convolution(gray, kernel)
        return np.stack([laplacian]*3, axis=-1).astype(np.uint8)  # Converte de volta para RGB
    else:
        return image

def segmentation_filter(image, threshold):
    gray = convert_to_gray(image)
    binary = (gray > threshold).astype(np.uint8) * 255
    return np.stack([binary]*3, axis=-1)

def binary_threshold(image, threshold):
    gray = convert_to_gray(image)
    binary = (gray > threshold).astype(np.uint8) * 255
    return np.stack([binary]*3, axis=-1)

def adaptive_threshold(image):
    gray = convert_to_gray(image)
    adaptive = gray > (gray.mean() - gray.std())
    return np.stack([(adaptive * 255).astype(np.uint8)]*3, axis=-1)

def morphology_filter(image, morph_type, kernel_type):
    kernel = create_morph_kernel(kernel_type)
    print(f"Kernel usado: \n{kernel}")  # Log para verificar o kernel
    gray = convert_to_gray(image)  # Converter para escala de cinza
    print(f"Imagem em escala de cinza: {gray.shape}")

    if morph_type == "Erosion":
        result = manual_erosion(gray, kernel)
        print("Erosão aplicada.")
    elif morph_type == "Dilation":
        result = manual_dilation(gray, kernel)
        print("Dilatação aplicada.")
    elif morph_type == "Opening":
        # Erosão seguida de dilatação
        eroded = manual_erosion(gray, kernel)
        result = manual_dilation(eroded, kernel)
        print("Abertura aplicada.")
    elif morph_type == "Closing":
        # Dilatação seguida de erosão
        dilated = manual_dilation(gray, kernel)
        result = manual_erosion(dilated, kernel)
        print("Fechamento aplicado.")
    else:
        print("Operação desconhecida.")
        return image

    # Converter o resultado para RGB para exibição
    return np.stack([result] * 3, axis=-1)

def manual_dilation(image, kernel):
    pad = kernel.shape[0] // 2
    h, w = image.shape
    output = np.zeros_like(image)
    padded = np.pad(image, pad, mode="constant", constant_values=0)  # Preencher bordas com preto
    for y in range(h):
        for x in range(w):
            region = padded[y:y + kernel.shape[0], x:x + kernel.shape[1]]
            output[y, x] = np.max(region[kernel == 1])  # Valor máximo onde o kernel é 1
    return output


def manual_erosion(image, kernel):
    pad = kernel.shape[0] // 2
    h, w = image.shape
    output = np.zeros_like(image)
    padded = np.pad(image, pad, mode="constant")
    for y in range(h):
        for x in range(w):
            region = padded[y:y + kernel.shape[0], x:x + kernel.shape[1]]
            output[y, x] = np.min(region[kernel == 1])
    return output

def manual_dilation(image, kernel):
    pad = kernel.shape[0] // 2
    h, w = image.shape
    output = np.zeros_like(image)
    padded = np.pad(image, pad, mode="constant")
    for y in range(h):
        for x in range(w):
            region = padded[y:y + kernel.shape[0], x:x + kernel.shape[1]]
            output[y, x] = np.max(region[kernel == 1])
    return output

def convert_to_gray(image):
    return (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(np.uint8)

def refresh_canvas():
    edited_image_canvas.delete("all")

def delayed_update(func):
    global delay_timer
    if delay_timer:
        root.after_cancel(delay_timer)
    delay_timer = root.after(500, func)

# Configuração da interface gráfica
root = tk.Tk()
root.geometry("1400x600")
root.title("Processador de Imagem")

img = None
delay_timer = None

menu = tk.Menu(root)
root.config(menu=menu)

file_menu = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Arquivo", menu=file_menu)
file_menu.add_command(label="Carregar Imagem", command=load_image)
file_menu.add_command(label="Sair", command=root.quit)

original_image_canvas = tk.Canvas(root, width=500, height=500, bg="#ddd")
original_image_canvas.grid(row=0, column=0, padx=10, pady=10)

edited_image_canvas = tk.Canvas(root, width=500, height=500, bg="#ddd")
edited_image_canvas.grid(row=0, column=1, padx=10, pady=10)

notebook = ttk.Notebook(root)
notebook.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

basic_filters_frame = ttk.Frame(notebook)
notebook.add(basic_filters_frame, text="Filtros Básicos")
filter_var = tk.StringVar(value="Low Pass")
filter_menu = tk.OptionMenu(
    basic_filters_frame,
    filter_var,
    "Low Pass", "High Pass", "Mean", "Binary Threshold", "Adaptive Threshold",
    command=lambda _: delayed_update(lambda: apply_filter(filter_var.get()))
)
filter_menu.pack(pady=5)
radius_scale = tk.Scale(basic_filters_frame, from_=1, to=10, label="Raio", orient="horizontal", command=lambda _: delayed_update(lambda: apply_filter(filter_var.get())))
radius_scale.pack(pady=5)
high_pass_var = tk.StringVar(value="Sobel")
high_pass_menu = tk.OptionMenu(basic_filters_frame, high_pass_var, "Sobel", "Laplacian")
high_pass_menu.pack(pady=5)

morphology_frame = ttk.Frame(notebook)
notebook.add(morphology_frame, text="Operações Morfológicas")
morph_type_var = tk.StringVar(value="Erosion")
morph_type_menu = tk.OptionMenu(
    morphology_frame,
    morph_type_var,
    "Erosion", "Dilation", "Opening", "Closing"
)
morph_type_menu.pack(pady=5)
kernel_type_var = tk.StringVar(value="Block 3x3")
kernel_type_menu = tk.OptionMenu(morphology_frame, kernel_type_var, "Block 3x3", "Cross 3x3")
kernel_type_menu.pack(pady=5)

segmentation_frame = ttk.Frame(notebook)
notebook.add(segmentation_frame, text="Segmentação")
threshold_scale = tk.Scale(segmentation_frame, from_=0, to=255, label="Limiar", orient="horizontal", command=lambda _: delayed_update(lambda: apply_filter("Segmentation")))
threshold_scale.pack(pady=5)

loading_label = tk.Label(root, text="", fg="red", font=("Arial", 12))
loading_label.grid(row=1, column=0, columnspan=3, pady=10)


def clear_filters():
    if img is not None:
        display_image(img, original=False)


clear_button = tk.Button(basic_filters_frame, text="Clear Filters", command=clear_filters)
clear_button.pack(pady=10)


root.mainloop()

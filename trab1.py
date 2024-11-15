import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

def load_image():
    global img_cv
    file_path = filedialog.askopenfilename()
    if file_path:
        img_cv = np.array(Image.open(file_path))  # Carrega a imagem usando PIL
        display_image(img_cv, original=True)  # Exibe a imagem original
        refresh_canvas()

def display_image(img, original=False):
    img_pil = Image.fromarray(img)
    
    # Obtém o tamanho da imagem original
    img_width, img_height = img_pil.size
    
    # Redimensiona a imagem para caber no canvas se for muito grande
    max_size = 500
    img_pil.thumbnail((max_size, max_size))  # Mantém a proporção
    img_tk = ImageTk.PhotoImage(img_pil)

    # Calcula a posição para centralizar a imagem dentro do canvas se for menor
    canvas_width, canvas_height = max_size, max_size
    x_offset = (canvas_width - img_pil.width) // 2
    y_offset = (canvas_height - img_pil.height) // 2

    if original:
        original_image_canvas.delete("all")  # Limpa a canvas
        original_image_canvas.image = img_tk  # Mantém a referência viva - garbage collection
        original_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)
    else:
        edited_image_canvas.delete("all")  # Limpa a canvas
        edited_image_canvas.image = img_tk
        edited_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)

def apply_filter(filter_type):
    if img_cv is None:
        return
    if filter_type == "gaussian":
        filtered_img = manual_gaussian_filter(img_cv, kernel_size=5, sigma=1.0)
    elif filter_type == "laplacian":
        gray = convert_to_gray(img_cv)
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        filtered_img = manual_convolution(gray, laplacian_kernel)
        filtered_img = np.clip(np.abs(filtered_img), 0, 255).astype(np.uint8)
        filtered_img = np.stack([filtered_img] * 3, axis=-1)  # Converter para 3 canais (RGB)
    elif filter_type == "sobel":
        gray = convert_to_gray(img_cv)
        sobelx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Kernel Sobel X
        sobely_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Kernel Sobel Y
        sobelx = manual_convolution(gray, sobelx_kernel)
        sobely = manual_convolution(gray, sobely_kernel)
        sobel = np.sqrt(sobelx**2 + sobely**2)  # Magnitude dos gradientes
        sobel = np.clip(sobel, 0, 255).astype(np.uint8)
        filtered_img = np.stack([sobel] * 3, axis=-1)  # Converter para 3 canais (RGB)
    elif filter_type == "mean":
        filtered_img = manual_mean_filter(img_cv, 15)

    display_image(filtered_img, original=False)  # Exibe a imagem editada

def manual_convolution(image, kernel):
    """Aplica convolução manualmente com um kernel dado."""
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape
    output = np.zeros_like(image)

    # Deslocamento para o centróide do kernel
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Padding (adição de bordas) para lidar com as bordas da imagem
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    # Realiza a convolução
    for i in range(img_height):
        for j in range(img_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]  # Região da imagem atual
            output[i, j] = np.sum(region * kernel)  # Multiplica elemento a elemento e soma
    return output

def manual_mean_filter(image, kernel_size):
    """Aplica um filtro de média (blur) manualmente a cada canal RGB."""
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    # Separar os canais R, G, B da imagem
    r_channel, g_channel, b_channel = image[..., 0], image[..., 1], image[..., 2]
    
    # Aplicar a convolução manualmente para cada canal
    r_filtered = manual_convolution(r_channel, kernel)
    g_filtered = manual_convolution(g_channel, kernel)
    b_filtered = manual_convolution(b_channel, kernel)
    
    # Recombinar os canais filtrados em uma imagem RGB
    filtered_img = np.stack([r_filtered, g_filtered, b_filtered], axis=-1)
    
    # Garantir que os valores estejam na faixa [0, 255]
    filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)
    
    return filtered_img

def manual_gaussian_filter(image, kernel_size, sigma):
    """Aplica filtro Gaussiano manualmente em cada canal RGB."""
    kernel = gaussian_kernel(kernel_size, sigma)
    return manual_mean_filter(image, kernel_size)

def gaussian_kernel(size, sigma):
    """Cria um kernel Gaussiano 2D."""
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

def convert_to_gray(image):
    """Converte uma imagem para tons de cinza manualmente."""
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])  # Conversão para escala de cinza

def refresh_canvas():
    edited_image_canvas.delete("all")  # Limpa a canvas para exibir a nova imagem

# Definindo a GUI
root = tk.Tk()
root.title("Image Processing - APP - Step 1")

# Define o tamanho da janela da aplicação 1200x800
root.geometry("1500x550")

# Define a cor de fundo da janela
root.config(bg="#2e2e2e")

img_cv = None

# Cria o menu da aplicação
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

# File menu
file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Load Image", command=load_image)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

# Filters menu
filters_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Filters", menu=filters_menu)
filters_menu.add_command(label="Gaussian Filter", command=lambda: apply_filter("gaussian"))
filters_menu.add_command(label="Laplacian Filter", command=lambda: apply_filter("laplacian"))
filters_menu.add_command(label="Sobel Filter", command=lambda: apply_filter("sobel"))
filters_menu.add_command(label="Mean Filter", command=lambda: apply_filter("mean"))

# Cria a canvas para a imagem original com borda (sem background)
original_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
original_image_canvas.grid(row=0, column=1, padx=20, pady=20)

# Cria a canvas para a imagem editada com borda (sem background)
edited_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
edited_image_canvas.grid(row=0, column=2, padx=20, pady=20)

# Cria os botões de filtro na interface em duas colunas
filter_frame = tk.Frame(root, bg="#2e2e2e")
filter_frame.grid(row=0, column=0, padx=10, pady=10)

# Botões para aplicar os filtros (coluna 1)
low_pass_button = tk.Button(filter_frame, text="Gaussian Filter", command=lambda: apply_filter("gaussian"), bg="#ff9999")
low_pass_button.grid(row=0, column=0, padx=10, pady=5)

high_pass_button = tk.Button(filter_frame, text="Laplacian Filter", command=lambda: apply_filter("laplacian"), bg="#ffcc99")
high_pass_button.grid(row=1, column=0, padx=10, pady=5)

# Botões para aplicar os filtros (coluna 2)
sobel_button = tk.Button(filter_frame, text="Sobel Filter", command=lambda: apply_filter("sobel"), bg="#99ccff")
sobel_button.grid(row=0, column=1, padx=10, pady=5)

mean_button = tk.Button(filter_frame, text="Mean Filter", command=lambda: apply_filter("mean"), bg="#99ff99")
mean_button.grid(row=1, column=1, padx=10, pady=5)

# Botões para operações morfológicas (coluna 1)
erosion_button = tk.Button(filter_frame, text="Erosion", command=lambda: apply_morphology("erosion"), bg="#ffb3b3")
erosion_button.grid(row=2, column=0, padx=10, pady=5)

dilation_button = tk.Button(filter_frame, text="Dilation", command=lambda: apply_morphology("dilation"), bg="#ffd9b3")
dilation_button.grid(row=3, column=0, padx=10, pady=5)

# Botões para operações morfológicas (coluna 2)
opening_button = tk.Button(filter_frame, text="Opening", command=lambda: apply_morphology("opening"), bg="#b3d9ff")
opening_button.grid(row=2, column=1, padx=10, pady=5)

closing_button = tk.Button(filter_frame, text="Closing", command=lambda: apply_morphology("closing"), bg="#b3ffb3")
closing_button.grid(row=3, column=1, padx=10, pady=5)

# Botões para segmentação (coluna 1)
binary_threshold_button = tk.Button(filter_frame, text="Binary Threshold", command=lambda: apply_segmentation("binary_threshold"), bg="#ff9999")
binary_threshold_button.grid(row=4, column=0, padx=10, pady=5)

adaptive_threshold_button = tk.Button(filter_frame, text="Adaptive Threshold", command=lambda: apply_segmentation("adaptive_threshold"), bg="#ffcc99")
adaptive_threshold_button.grid(row=4, column=1, padx=10, pady=5)


def apply_morphology(operation):
    """
    Aplica uma operação morfológica (erosão, dilatação, abertura ou fechamento).
    """
    if img_cv is None:
        return

    gray = convert_to_gray(img_cv)
    binary = (gray > 128).astype(np.uint8) * 255  # Converte para binário

    if operation == "erosion":
        result = erosion(binary, kernel_size=3)
    elif operation == "dilation":
        result = dilation(binary, kernel_size=3)
    elif operation == "opening":
        result = dilation(erosion(binary, kernel_size=3), kernel_size=3)
    elif operation == "closing":
        result = erosion(dilation(binary, kernel_size=3), kernel_size=3)

    # Converte o resultado para RGB e exibe
    display_image(np.stack([result] * 3, axis=-1), original=False)


def erosion(image, kernel_size):
    """
    Aplica a operação de erosão em uma imagem binária ou tons de cinza.

    Parâmetros:
        image (numpy.ndarray): Imagem de entrada (2D).
        kernel_size (int): Tamanho do kernel (deve ser ímpar).

    Retorno:
        numpy.ndarray: Imagem erodida.
    """
    # Verifica se o tamanho do kernel é ímpar
    if kernel_size % 2 == 0:
        raise ValueError("O tamanho do kernel deve ser ímpar.")

    # Criação do kernel (estrutura retangular)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Adiciona padding à imagem
    pad = kernel_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

    # Inicializa a imagem de saída
    output = np.zeros_like(image)

    # Aplica erosão manualmente
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extrai a região correspondente ao kernel
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            # Calcula o valor mínimo na região
            output[i, j] = np.min(region * kernel)

    return output


def dilation(image, kernel_size):
    """
    Aplica a operação de dilatação em uma imagem binária ou tons de cinza.

    Parâmetros:
        image (numpy.ndarray): Imagem de entrada (2D).
        kernel_size (int): Tamanho do kernel (deve ser ímpar).

    Retorno:
        numpy.ndarray: Imagem dilatada.
    """
    # Verifica se o tamanho do kernel é ímpar
    if kernel_size % 2 == 0:
        raise ValueError("O tamanho do kernel deve ser ímpar.")

    # Criação do kernel (estrutura retangular)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Adiciona padding à imagem
    pad = kernel_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

    # Inicializa a imagem de saída
    output = np.zeros_like(image)

    # Aplica dilatação manualmente
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extrai a região correspondente ao kernel
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            # Calcula o valor máximo na região
            output[i, j] = np.max(region)

    return output


def adaptive_threshold(image):
    """Aplica limiarização adaptativa (média da vizinhança)."""
    kernel_size = 11  # Tamanho da vizinhança
    offset = 15  # Subtração para ajustar o limiar
    padded_image = np.pad(image, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='constant', constant_values=0)
    output = np.zeros_like(image, dtype=np.uint8)  # Garante que o tipo seja uint8

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            local_threshold = np.mean(region) - offset
            output[i, j] = 255 if image[i, j] > local_threshold else 0
    return output

def apply_segmentation(method):
    if img_cv is None:
        return

    gray = convert_to_gray(img_cv)

    if method == "binary_threshold":
        threshold = 128
        segmented = (gray > threshold).astype(np.uint8) * 255
    elif method == "adaptive_threshold":
        segmented = adaptive_threshold(gray)

    segmented = np.clip(segmented, 0, 255).astype(np.uint8)  # Garante o formato correto
    display_image(np.stack([segmented] * 3, axis=-1), original=False)  # Exibe a segmentação

# Expande a interface gráfica
morphology_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Morphology", menu=morphology_menu)
morphology_menu.add_command(label="Erosion", command=lambda: apply_morphology("erosion"))
morphology_menu.add_command(label="Dilation", command=lambda: apply_morphology("dilation"))
morphology_menu.add_command(label="Opening", command=lambda: apply_morphology("opening"))
morphology_menu.add_command(label="Closing", command=lambda: apply_morphology("closing"))

segmentation_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Segmentation", menu=segmentation_menu)
segmentation_menu.add_command(label="Binary Threshold", command=lambda: apply_segmentation("binary_threshold"))
segmentation_menu.add_command(label="Adaptive Threshold", command=lambda: apply_segmentation("adaptive_threshold"))


root.mainloop()
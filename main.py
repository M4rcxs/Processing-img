import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

def load_image():
    global img_cv
    file_path = filedialog.askopenfilename()
    if file_path:
        img_cv = cv2.imread(file_path)
        display_image(img_cv, original=True)  # Exibe a imagem original
        refresh_canvas()

def display_image(img, original=False):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
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
    if filter_type == "low_pass":
        filtered_img = cv2.GaussianBlur(img_cv, (15, 15), 0)
    elif filter_type == "high_pass":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        filtered_img = cv2.Laplacian(gray, cv2.CV_64F)
        filtered_img = cv2.convertScaleAbs(filtered_img)
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    elif filter_type == "sobel":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Derivada X
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Derivada Y
        sobel = cv2.magnitude(sobelx, sobely)  # Calcula a magnitude dos gradientes
        sobel = cv2.convertScaleAbs(sobel)  # Converte para uma imagem de 8 bits
        filtered_img = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)  # Converte para 3 canais
    elif filter_type == "mean":
        filtered_img = cv2.blur(img_cv, (15, 15))  # Filtro de média com kernel 15x15
    display_image(filtered_img, original=False)  # Exibe a imagem editada

def refresh_canvas():
    edited_image_canvas.delete("all")  # Limpa a canvas para exibir a nova imagem

# Definindo a GUI
root = tk.Tk()
root.title("Image Processing App")

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
filters_menu.add_command(label="Low Pass Filter", command=lambda: apply_filter("low_pass"))
filters_menu.add_command(label="High Pass Filter", command=lambda: apply_filter("high_pass"))
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
low_pass_button = tk.Button(filter_frame, text="Low Pass Filter", command=lambda: apply_filter("low_pass"), bg="#ff9999")
low_pass_button.grid(row=0, column=0, padx=10, pady=5)

high_pass_button = tk.Button(filter_frame, text="High Pass Filter", command=lambda: apply_filter("high_pass"), bg="#ffcc99")
high_pass_button.grid(row=1, column=0, padx=10, pady=5)

# Botões para aplicar os filtros (coluna 2)
sobel_button = tk.Button(filter_frame, text="Sobel Filter", command=lambda: apply_filter("sobel"), bg="#99ccff")
sobel_button.grid(row=0, column=1, padx=10, pady=5)

mean_button = tk.Button(filter_frame, text="Mean Filter", command=lambda: apply_filter("mean"), bg="#99ff99")
mean_button.grid(row=1, column=1, padx=10, pady=5)

root.mainloop()

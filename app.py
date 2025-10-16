# app.py

import tkinter as tk
from tkinter import messagebox
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- Global variables ---
model = None
cv = None
ps = PorterStemmer()

# --- Text Preprocessing ---
def preprocess_text(text):
    try:
        stops = stopwords.words('english')
    except:
        nltk.download('stopwords')
        stops = stopwords.words('english')
        
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if not word in stops]
    return ' '.join(review)

# --- Prediction ---
def classify_email():
    global text_entry, result_label
    email_text = text_entry.get("1.0", tk.END)
    
    if len(email_text.strip()) < 1:
        messagebox.showwarning("Input Error", "Please enter text to classify.")
        return

    processed_text = preprocess_text(email_text)
    vectorized_text = cv.transform([processed_text]).toarray()
    prediction = model.predict(vectorized_text)
    
    if prediction[0] == 1:
        result_label.config(text="Result: This is SPAM!", fg="red")
    else:
        result_label.config(text="Result: This is NOT SPAM.", fg="green")

# --- UI Navigation ---
def load_model_and_show_main_ui(dataset_choice):
    global model, cv
    model_path = f"MNB_d{dataset_choice}.pkl"
    vectorizer_path = f"vectorizer_d{dataset_choice}.pkl"

    try:
        with open(model_path, "rb") as f_model, open(vectorizer_path, "rb") as f_vec:
            model = pickle.load(f_model)
            cv = pickle.load(f_vec)
        
        for widget in root.winfo_children():
            widget.destroy()
        setup_main_ui()

    except FileNotFoundError:
        messagebox.showerror("Error", f"Files not found!\nMake sure '{model_path}' and '{vectorizer_path}' exist.")

def show_selection_screen():
    for widget in root.winfo_children():
        widget.destroy()
    setup_selection_ui()

# --- UI Setup ---
def setup_main_ui():
    global text_entry, result_label
    root.title("Email Spam Classifier")
    
    # --- NEW: Use frames for better layout ---
    # A top frame for all content except the back button
    top_frame = tk.Frame(root, bg="#f0f0f0")
    top_frame.pack(fill="both", expand=True, padx=10, pady=5)

    # A bottom frame just for the back button
    bottom_frame = tk.Frame(root, bg="#f0f0f0")
    bottom_frame.pack(fill="x", side="bottom")

    # --- Add widgets to the top_frame ---
    title_label = tk.Label(top_frame, text="Email Spam Classifier", font=("Helvetica", 18, "bold"), bg="#f0f0f0")
    title_label.pack(pady=10)

    instruction_label = tk.Label(top_frame, text="Enter the email content below:", font=("Helvetica", 11), bg="#f0f0f0")
    instruction_label.pack(pady=5)

    text_entry = tk.Text(top_frame, height=10, width=55, font=("Helvetica", 10), relief=tk.SOLID, borderwidth=1)
    text_entry.pack(pady=10, padx=10)

    classify_button = tk.Button(top_frame, text="Classify Email", command=classify_email, font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white", relief=tk.RAISED)
    classify_button.pack(pady=10)

    result_label = tk.Label(top_frame, text="Result: ", font=("Helvetica", 14, "bold"), bg="#f0f0f0")
    result_label.pack(pady=10)

    # --- Add the back button to the bottom_frame ---
    back_button = tk.Button(
        bottom_frame,
        text="< Back to Dataset Selection",
        command=show_selection_screen,
        font=("Helvetica", 10),
        bg="#6c757d",
        fg="white"
    )
    back_button.pack(pady=15) # Add padding to center it vertically

def setup_selection_ui():
    root.title("Select Dataset")
    
    selection_frame = tk.Frame(root, bg="#f0f0f0")
    selection_frame.pack(expand=True, fill="both")

    title_label = tk.Label(selection_frame, text="Select a Dataset", font=("Helvetica", 18, "bold"), bg="#f0f0f0")
    title_label.pack(pady=20)

    instruction_label = tk.Label(selection_frame, text="Which dataset would you like to use?", font=("Helvetica", 11), bg="#f0f0f0")
    instruction_label.pack(pady=10)
    
    dataset1_button = tk.Button(selection_frame, text="Use Dataset 1", command=lambda: load_model_and_show_main_ui(1), font=("Helvetica", 12, "bold"), bg="#007BFF", fg="white", width=20)
    dataset1_button.pack(pady=10)
    
    dataset2_button = tk.Button(selection_frame, text="Use Dataset 2", command=lambda: load_model_and_show_main_ui(2), font=("Helvetica", 12, "bold"), bg="#28A745", fg="white", width=20)
    dataset2_button.pack(pady=10)

# --- Main Application Window ---
if __name__ == "__main__":
    root = tk.Tk()
    # Increased the height one last time for comfort
    root.geometry("500x500") 
    root.resizable(False, False)
    root.configure(bg="#f0f0f0")

    setup_selection_ui()
    root.mainloop()
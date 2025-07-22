import threading
import webview
from pywebio.input import input, actions
from pywebio.output import put_markdown, put_button, clear, put_scope, clear_scope
from pywebio import start_server
from generator import generate_poem, generate_acrostic, get_model

def generate_poem(start_char, is_acrostic):
    model, char2idx, idx2char = get_model()
    if not start_char:
        return "Please enter a character or word."
    if is_acrostic:
        return generate_acrostic(model, char2idx, idx2char, start_char, device='cpu')
    else:
        return generate_poem(model, char2idx, idx2char, start_text=start_char, device='cpu')
   
def poem_ui():
    while True:
        clear()
        start_char = input("Input Starting Text:", type="text")
        mode = actions(label="Select Mode", buttons=["Normal Generation", "Acrostic"])
        result = generate_poem(start_char, is_acrostic=(mode == "Acrostic"))
        scope_name = "result_scope"
        clear_scope(scope_name)
        put_scope(scope_name)
        put_markdown(f"### Generated Result:\n\n{result}")
        btn_clicked = None
        def on_click():
            nonlocal btn_clicked
            btn_clicked = True
        put_button("Replay (takes a few seconds)", onclick=on_click)
        while not btn_clicked:
            pass

       
def start_pywebio():
    start_server(poem_ui, port=8080, debug=True)

if __name__ == "__main__":

    threading.Thread(target=start_pywebio, daemon=True).start()
    webview.create_window("Poem Generator", "http://localhost:8080", width=600, height=500)
    webview.start()
import tkinter as tk
from tkinter import messagebox


class JapanHelper:
    @staticmethod
    def set_lang(lang, form):
        # Not applicable in Python, as setting language at runtime is not common
        pass

    @staticmethod
    def app_lang(control, resources):
        if isinstance(control, tk.Menu):
            # Set resources for MenuStrip
            resources.update({control._name: control.cget('text')})
            for c in control.winfo_children():
                JapanHelper.app_lang(c, resources)
        elif isinstance(control, tk.Widget):
            # Set resources for other widgets
            resources.update({control._name: control.cget('text')})
            for c in control.winfo_children():
                JapanHelper.app_lang(c, resources)

    @staticmethod
    def traverse_menus(item, resources):
        if isinstance(item, tk.Menu):
            # Set resources for ToolStripMenuItem
            resources.update({item._name: item.cget('text')})
            for c in item.winfo_children():
                JapanHelper.traverse_menus(c, resources)


if __name__ == "__main__":
    # Example usage of JapanHelper class
    app = tk.Tk()

    # Create a simple menu and menu items
    menu_bar = tk.Menu(app)
    file_menu = tk.Menu(menu_bar, tearoff=0)
    file_menu.add_command(label="Open", command=lambda: messagebox.showinfo("Info", "Open clicked"))
    file_menu.add_command(label="Save", command=lambda: messagebox.showinfo("Info", "Save clicked"))
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=app.quit)

    menu_bar.add_cascade(label="File", menu=file_menu)
    app.config(menu=menu_bar)

    # Apply language settings using JapanHelper
    lang = "zh-CN"  # Change to "en-US" for English
    JapanHelper.set_lang(lang, app)

    app.mainloop()

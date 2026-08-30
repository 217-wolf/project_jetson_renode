
"""Prosty GUI do uruchamiania i monitorowania procesu zbierania embeddingów.

Interfejs ma trzy karty:
 - Log: podgląd ostatnich wpisów CSV
 - Summary: podsumowanie liczby embeddingów na klasę
 - Controls: start/stop/restart procesu oraz przycisk otwierający folder z zapisami
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import sys, os
import threading
from collect_embeddings import EmbeddingCollector
from pathlib import Path
from reid_tracker import EmbeddingStore


class CameraGUI:
    def __init__(self, master):
        self.master = master
        master.title('Camera Embedding Collector')
        self.store = EmbeddingStore(Path('patterns_database'))

        self.collector = None
        self._collector_thread = None

        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill='both', expand=True)

        # Log tab
        self.tab_log = ttk.Frame(self.notebook)
        self.txt_log = scrolledtext.ScrolledText(self.tab_log, height=20)
        self.txt_log.pack(fill='both', expand=True)
        self.notebook.add(self.tab_log, text='Log')

        # Summary tab
        self.tab_summary = ttk.Frame(self.notebook)
        self.lbl_summary = tk.Label(self.tab_summary, text='', justify='left', anchor='nw')
        self.lbl_summary.pack(fill='both', expand=True, padx=8, pady=8)
        self.notebook.add(self.tab_summary, text='Summary')

        # Controls
        self.tab_ctrl = ttk.Frame(self.notebook)
        self.btn_start = ttk.Button(self.tab_ctrl, text='Start', command=self.start_process)
        self.btn_stop = ttk.Button(self.tab_ctrl, text='Stop', command=self.stop_process)
        self.btn_restart = ttk.Button(self.tab_ctrl, text='Restart', command=self.restart_process)
        self.btn_open = ttk.Button(self.tab_ctrl, text='Open Folder', command=self.open_folder)
        self.btn_start.pack(side='left', padx=6, pady=10)
        self.btn_stop.pack(side='left', padx=6, pady=10)
        self.btn_restart.pack(side='left', padx=6, pady=10)
        self.btn_open.pack(side='left', padx=6, pady=10)
        self.notebook.add(self.tab_ctrl, text='Controls')

        # Schedule updates
        self.update_interval = 1000
        self._schedule_update()

    def start_process(self):
        if self.collector is not None:
            messagebox.showinfo('Info', 'Proces już działa')
            return
        self.collector = EmbeddingCollector()
        self._collector_thread = threading.Thread(target=self.collector.start, kwargs={'threaded':False}, daemon=True)
        self._collector_thread.start()
        self._append_log('Started embedding collector in thread\n')

    def stop_process(self):
        if self.collector is None:
            messagebox.showinfo('Info', 'Brak uruchomionego procesu')
            return
        try:
            self.collector.stop()
            if self._collector_thread:
                self._collector_thread.join(timeout=5)
        finally:
            self.collector = None
            self._collector_thread = None
            self._append_log('Stopped embedding collector\n')

    def restart_process(self):
        self.stop_process()
        self.start_process()

    def open_folder(self):
        folder = self.store.reid_root.resolve()
        try:
            if sys.platform == 'win32':
                os.startfile(str(folder))
            else:
                # best-effort open
                os.system(f'xdg-open "{str(folder)}"')
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def _append_log(self, text: str):
        self.txt_log.insert('end', text)
        self.txt_log.see('end')

    def _refresh_from_storage(self):
        summary = self.store.get_summary()
        self.txt_log.delete('1.0', 'end')
        for class_name, counts in sorted(summary.items()):
            self.txt_log.insert(
                'end',
                f"{class_name}: {counts['identities']} tożsamości, {counts['gallery_samples']} widoków galerii\n",
            )

        text = '\n'.join(
            f"{class_name}: {counts['identities']} tożsamości, {counts['gallery_samples']} widoków"
            for class_name, counts in sorted(summary.items())
        )
        self.lbl_summary.config(text=text)

        # Check collector status
        if self.collector is not None:
            # collector runs in thread; no returncode, but we track presence
            pass

    def _schedule_update(self):
        try:
            self._refresh_from_storage()
        finally:
            self.master.after(self.update_interval, self._schedule_update)


def run():
    root = tk.Tk()
    app = CameraGUI(root)
    root.mainloop()


if __name__ == '__main__':
    run()

"""
demo.py - Giao dien cua so cho BKERW Toolkit.

Co 2 che do:
  1) Tien xu ly du lieu   -> TCGA_analyzer, Co-expression, disease_specific_ontologies
                              da noi voi script that (chay nen, khong lam dong giao dien).
                              seed_set, buildSimilarityMatrix con "chua trien khai".
  2) Chay thuat toan       -> chi co BKERW, nguoi dung chon/nhap ten Experiment,
                               thuat toan tu doc phan con lai tu config (Hydra)

Cach chay:
    python demo.py

Yeu cau: Python 3.8+, tkinter (co san trong Python chuan), file demo.py (Hydra)
nam cung thu muc voi demo.py.
"""

import os
import sys
import glob
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox

# ----------------------------------------------------------------------------
# Cau hinh duong dan (GIU NGUYEN - khong doi bat ky gia tri nao o day)
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(BASE_DIR, "demo.py")                       # script Hydra chay thuat toan
EXPERIMENT_CONFIG_DIR = os.path.join(BASE_DIR, "config", "experiment")  # noi chua config tung experiment

# Method dung cho BKERW (ban goc, khong dung gene similarity).
# Neu cau truc config cua ban khac, chinh lai gia tri nay.
BKERW_METHOD_VALUE = "experimental_gs"
PREPROCESS_DIR = os.path.join(BASE_DIR, "data_preprocessing")

# ----------------------------------------------------------------------------
# Bang mau giao dien
# ----------------------------------------------------------------------------
COLOR_BG = "#f4f7f9"
COLOR_PRIMARY = "#0f766e"      # teal - mau chinh (khoa hoc/sinh hoc)
COLOR_PRIMARY_DARK = "#0b5a54"
COLOR_ACCENT = "#0ea5e9"       # xanh duong - nut phu
COLOR_TEXT = "#1e293b"
COLOR_MUTED = "#64748b"
COLOR_SUCCESS = "#15803d"
COLOR_ERROR = "#b91c1c"
COLOR_CARD = "#ffffff"
COLOR_BORDER = "#dbe3e8"


def get_available_experiments():
    """Quet thu muc config/experiment de lay danh sach ten experiment co san (*.yaml)."""
    if not os.path.isdir(EXPERIMENT_CONFIG_DIR):
        return []
    files = glob.glob(os.path.join(EXPERIMENT_CONFIG_DIR, "*.yaml"))
    return sorted(os.path.splitext(os.path.basename(f))[0] for f in files)


# ----------------------------------------------------------------------------
# Cac ham tien xu ly dung chung
# (Chuyen ra cap module vi khong dung `self` - day la nguyen nhan gay
#  NameError trong ban goc khi chung bi dat thut vao trong class ma thieu self)
# ----------------------------------------------------------------------------
def run_preprocessing(script_name, args):
    script = os.path.join(PREPROCESS_DIR, script_name)
    cmd = [sys.executable, script] + args
    return subprocess.run(
        cmd,
        cwd=PREPROCESS_DIR,
        capture_output=True,
        text=True,
    )


def build_dataset_paths(dataset):
    return {
        "gdc": f"../data_raw/sample_sheets/{dataset}.tsv",
        "manifest": f"../data_raw/manifests/manifest_{dataset}.txt",
        "rna_dir": f"../data_raw/data/{dataset.replace('-', '_')}/",
        "output_dir": f"../data_set_news/{dataset.replace('-', '_')}/",
        "tumor": f"../data_set_news/{dataset.replace('-', '_')}/{dataset}__tumor.tsv",
        "control": f"../data_set_news/{dataset.replace('-', '_')}/{dataset}__control.tsv",
        "de": f"../data_set/differentially_expressed_genes/{dataset}_de_genes.tsv",
        "co": f"../data_set/co-expression_networks/{dataset}__co_expression__t_70%.tsv",
        "seed": f"../data_set/seed_set/{dataset}_seed.txt",
        "ontology": "../data_set/ontology_network/ontology_network.tsv",
        "disease": f"../data_set/disease_specific_ontologies/{dataset}_disease_ontologies.txt",
    }

DATASET_PARAMS = {
    "kirc": {
        "restart_prob": 0.8,
        "alpha": 0.9,
        "beta": 0.5,
    },
    "brca": {
        "restart_prob": 0.9,
        "alpha": 0.4,
        "beta": 0.6,
    },
    "luad": {
        "restart_prob": 0.85,
        "alpha": 0.5,
        "beta": 0.5,
    },
    "thca": {
    "restart_prob": 0.85,
    "alpha": 0.9,
    "beta": 0.7,
    },
    "stad": {
        "restart_prob": 0.9,
        "alpha": 0.9,
        "beta": 0.6,
    },
    "lihc": {
        "restart_prob": 0.9,
        "alpha": 0.9,
        "beta": 0.2,
    },
    "coad": {
        "restart_prob": 0.9,
        "alpha": 0.8,
        "beta": 0.9,
    },
    "chol": {
    "restart_prob": 0.9,
    "alpha": 0.9,
    "beta": 0.5,
    },
}

DEFAULT_PARAMS = {
    "restart_prob": 0.9,
    "alpha": 0.5,
    "beta": 0.5,
}
# ----------------------------------------------------------------------------
# Thiet lap style dung chung
# ----------------------------------------------------------------------------
def setup_style(root):
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    root.configure(bg=COLOR_BG)

    style.configure("TFrame", background=COLOR_BG)
    style.configure("Card.TFrame", background=COLOR_CARD)

    style.configure("Title.TLabel", background=COLOR_BG, foreground=COLOR_PRIMARY_DARK,
                     font=("Segoe UI", 21, "bold"))
    style.configure("Subtitle.TLabel", background=COLOR_BG, foreground=COLOR_MUTED,
                     font=("Segoe UI", 11))
    style.configure("Heading.TLabel", background=COLOR_BG, foreground=COLOR_TEXT,
                     font=("Segoe UI", 15, "bold"))
    style.configure("Body.TLabel", background=COLOR_BG, foreground=COLOR_TEXT,
                     font=("Segoe UI", 10))
    style.configure("Muted.TLabel", background=COLOR_BG, foreground=COLOR_MUTED,
                     font=("Segoe UI", 9))
    style.configure("Status.TLabel", background=COLOR_BG, foreground=COLOR_ACCENT,
                     font=("Segoe UI", 10, "bold"))
    style.configure("Success.TLabel", background=COLOR_BG, foreground=COLOR_SUCCESS,
                     font=("Segoe UI", 10, "bold"))
    style.configure("Error.TLabel", background=COLOR_BG, foreground=COLOR_ERROR,
                     font=("Segoe UI", 10, "bold"))

    style.configure("Primary.TButton", font=("Segoe UI", 11, "bold"),
                     foreground="white", background=COLOR_PRIMARY,
                     padding=10, borderwidth=0)
    style.map("Primary.TButton",
              background=[("active", COLOR_PRIMARY_DARK), ("disabled", "#94a3b8")])

    style.configure("Step.TButton", font=("Segoe UI", 10), foreground=COLOR_TEXT,
                     background="#e8eef0", padding=8, borderwidth=0)
    style.map("Step.TButton",
              background=[("active", "#d7e2e5"), ("disabled", "#f1f5f9")])

    style.configure("Ghost.TButton", font=("Segoe UI", 10), foreground=COLOR_MUTED,
                     background=COLOR_BG, padding=6, borderwidth=0)
    style.map("Ghost.TButton", foreground=[("active", COLOR_PRIMARY_DARK)])

    style.configure("TCombobox", padding=6)
    style.configure("TProgressbar", troughcolor="#e2e8f0", background=COLOR_ACCENT,
                     thickness=8)

    return style


# ----------------------------------------------------------------------------
# Ung dung chinh (dieu huong nhieu trang tren cung 1 cua so)
# ----------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BKERW Toolkit")
        self.geometry("640x540")
        self.minsize(640, 540)
        self.resizable(False, False)

        setup_style(self)

        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        self.frames = {}
        for PageClass in (HomePage, PreprocessPage, AlgorithmPage):
            frame = PageClass(container, self)
            self.frames[PageClass] = frame
            frame.place(relwidth=1, relheight=1)

        self.show_frame(HomePage)
        self._center_window()

    def _center_window(self):
        self.update_idletasks()
        w, h = 640, 540
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        x, y = (sw - w) // 2, (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    def show_frame(self, page_class):
        frame = self.frames[page_class]
        if hasattr(frame, "on_show"):
            frame.on_show()
        frame.tkraise()


class HomePage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, style="TFrame")

        ttk.Label(self, text="🧬 BKERW Toolkit", style="Title.TLabel").pack(pady=(60, 6))
        ttk.Label(
            self,
            text="Biological Knowledge Embedding + Random Walk\nCong cu ho tro nghien cuu",
            style="Subtitle.TLabel", justify="center"
        ).pack(pady=(0, 40))

        card = ttk.Frame(self, style="TFrame")
        card.pack()

        ttk.Button(
            card, text="🧪   Tien xu ly du lieu", width=34, style="Primary.TButton",
            command=lambda: controller.show_frame(PreprocessPage)
        ).pack(pady=8, ipady=4)

        ttk.Button(
            card, text="⚙️   Chay thuat toan (BKERW)", width=34, style="Primary.TButton",
            command=lambda: controller.show_frame(AlgorithmPage)
        ).pack(pady=8, ipady=4)

        ttk.Label(self, text="HUST · Do an tot nghiep", style="Muted.TLabel").pack(side="bottom", pady=20)


class PreprocessPage(ttk.Frame):
    """Giao dien tien xu ly du lieu.

    3 buoc dau (TCGA_analyzer, Co-expression, disease_specific_ontologies) da
    noi voi script that va CHAY NEN (khong lam dong giao dien). 2 buoc con lai
    (seed_set, buildSimilarityMatrix) chua co code xu ly.
    """

    STEP_DEFS = [
        ("seed_set", "run_seed_set"),
        ("TCGA_analyzer", "run_tcga_analyzer"),
        ("Co-expression va Differentially_expressed", "run_co_expression"),
        ("disease_specific_ontologies", "run_disease_ontology"),
        ("buildSimilarityMatrix", "run_build_similarity_matrix"),
    ]

    def __init__(self, parent, controller):
        super().__init__(parent, style="TFrame")
        self.controller = controller
        self.step_buttons = []
        self._is_running = False

        ttk.Label(self, text="Tien xu ly du lieu", style="Heading.TLabel").pack(pady=(24, 4))
        ttk.Label(self, text="Chon dataset roi chay tung buoc theo thu tu",
                   style="Muted.TLabel").pack(pady=(0, 16))

        steps_frame = ttk.Frame(self, style="TFrame")
        steps_frame.pack(pady=4)

        handlers = {
            "run_seed_set": self.run_seed_set,
            "run_tcga_analyzer": self.run_tcga_analyzer,
            "run_co_expression": self.run_co_expression,
            "run_disease_ontology": self.run_disease_ontology,
            "run_build_similarity_matrix": self.run_build_similarity_matrix,
        }

        for i, (label, handler_name) in enumerate(self.STEP_DEFS, start=1):
            if handler_name:
                cmd = handlers[handler_name]
            else:
                cmd = lambda name=label: self.not_implemented(name)

            btn = ttk.Button(steps_frame, text=f"{i}.  {label}", width=46,
                              style="Step.TButton", command=cmd)
            btn.pack(pady=4, ipady=3)
            self.step_buttons.append(btn)

        dataset_frame = ttk.Frame(self, style="TFrame")
        dataset_frame.pack(pady=(18, 4))

        ttk.Label(dataset_frame, text="Dataset:", style="Body.TLabel").grid(row=0, column=0, padx=(0, 8))

        self.dataset_var = tk.StringVar()
        datasets = ["TCGA-CHOL", "TCGA-HNSC", "TCGA-ESCA", "TCGA-LIHC", "TCGA-KIRC", "TCGA-THCA", "TCGA-STAD", "TCGA-COAD", "TCGA-KICH"]
        self.dataset_combo = ttk.Combobox(
            dataset_frame, textvariable=self.dataset_var, values=datasets,
            width=22, state="readonly"
        )
        self.dataset_combo.grid(row=0, column=1)
        self.dataset_combo.current(0)

        self.progress = ttk.Progressbar(self, mode="indeterminate", length=380)
        self.status_var = tk.StringVar(value="San sang.")
        self.status_label = ttk.Label(self, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.pack(pady=(18, 4))
        self.progress.pack(pady=(0, 10))
        self.progress.pack_forget()  # an cho den khi co tien trinh chay

        ttk.Button(self, text="←  Quay lai", style="Ghost.TButton",
                   command=lambda: controller.show_frame(HomePage)).pack(pady=10)

    # -- helpers --------------------------------------------------------
    def _set_running(self, running, message=""):
        self._is_running = running
        state = "disabled" if running else "normal"
        for btn in self.step_buttons:
            btn.config(state=state)
        if running:
            self.status_label.config(style="Status.TLabel")
            self.status_var.set(message or "Dang xu ly...")
            self.progress.pack(pady=(0, 10))
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.pack_forget()

    def _run_step_async(self, target, *args):
        if self._is_running:
            return
        self._set_running(True, "Dang chay, vui long doi...")
        threading.Thread(target=self._worker, args=(target, args), daemon=True).start()

    def _worker(self, target, args):
        try:
            result = target(*args)
            success = result.returncode == 0
            err = result.stderr
        except Exception as e:
            success = False
            err = str(e)
        self.after(0, self._on_step_finished, success, err)

    def _on_step_finished(self, success, err):
        self._set_running(False)
        if success:
            self.status_label.config(style="Success.TLabel")
            self.status_var.set("Hoan thanh.")
            messagebox.showinfo("OK", "Hoan thanh.")
        else:
            self.status_label.config(style="Error.TLabel")
            self.status_var.set("Co loi xay ra.")
            messagebox.showerror("Loi", err or "Khong ro loi.")

    # -- cac buoc xu ly ---------------------------------------------------
    def run_tcga_analyzer(self):
        dataset = self.dataset_var.get()
        p = build_dataset_paths(dataset)
        self._run_step_async(
            run_preprocessing,
            "TCGA_analyzer.py",
            ["-gdc", p["gdc"], "-m", p["manifest"], "-rna_dir", p["rna_dir"], "-o", p["output_dir"]],
        )

    def run_seed_set(self):
        if self._is_running:
            return

        dataset = self.dataset_var.get()
        cancer = dataset.replace("TCGA-", "")

        self._set_running(True, "Đang tạo seed gene...")

        def worker():
            script = os.path.join(BASE_DIR, "generate_seed.py")

            cmd = [
                sys.executable,
                script,
                "--cancer",
                cancer,
                "--threshold",
                "1",
            ]

            try:
                result = subprocess.run(
                    cmd,
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True,
                )
                success = result.returncode == 0
                output_log = (result.stdout or "") + "\n" + (result.stderr or "")
            except Exception as e:
                success = False
                output_log = str(e)

            self.after(0, self._on_step_finished, success, output_log)

        threading.Thread(target=worker, daemon=True).start()

    def run_co_expression(self):
        dataset = self.dataset_var.get()
        p = build_dataset_paths(dataset)
        self._run_step_async(
            run_preprocessing,
            "compute_co_expression_and_de_genes.py",
            ["-T", p["tumor"], "-C", p["control"], "-de", p["de"], "-co", p["co"]],
        )

    def run_disease_ontology(self):
        dataset = self.dataset_var.get()
        p = build_dataset_paths(dataset)
        self._run_step_async(
            run_preprocessing,
            "compute_disease_specific_ontologies.py",
            ["-s", p["seed"], "-a", p["ontology"], "-o", p["disease"]],
        )

    def run_build_similarity_matrix(self):
        if self._is_running:
            return

        dataset = self.dataset_var.get()
        cancer = dataset.replace("TCGA-", "")

        self._set_running(True, "Đang xây dựng Gene Similarity Matrix...")

        def worker():
            script = os.path.join(BASE_DIR, "buildMatrixGeneSim.py")

            cmd = [
                sys.executable,
                script,
                "mode=3",
                f"experiment.name={cancer}",
            ]

            try:
                result = subprocess.run(
                    cmd,
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True,
                )

                success = result.returncode == 0
                output_log = (result.stdout or "") + "\n" + (result.stderr or "")

            except Exception as e:
                success = False
                output_log = str(e)

            self.after(0, self._on_step_finished, success, output_log)

        threading.Thread(target=worker, daemon=True).start()
        
    def not_implemented(self, step_name):
        messagebox.showinfo(
            "Chua trien khai",
            f"Chuc nang '{step_name}' chua duoc nhung code xu ly.\n"
            "Hay bo sung logic sau."
        )


class AlgorithmPage(ttk.Frame):
    """Giao dien chay thuat toan BKERW.
    Nguoi dung chi can chon/nhap ten Experiment; cac tham so con lai
    (ppi, co-expression, disease ontology, restart_prob, alpha, beta, k_list...)
    duoc thuat toan tu doc tu config Hydra (config/experiment/<name>.yaml,
    config/paths, config/params, config/evaluation...).
    """

    def __init__(self, parent, controller):
        super().__init__(parent, style="TFrame")

        ttk.Label(self, text="Chay thuat toan BKERW", style="Heading.TLabel").pack(pady=(28, 16))

        form = ttk.Frame(self, style="TFrame")
        form.pack(pady=6)

        ttk.Label(form, text="Experiment:", style="Body.TLabel").grid(row=0, column=0, sticky="w", padx=6, pady=6)

        self.experiment_var = tk.StringVar()
        self.combo = ttk.Combobox(form, textvariable=self.experiment_var, width=38)
        self.combo.grid(row=0, column=1, padx=6, pady=6)

        ttk.Label(
            form,
            text="(Chon tu danh sach co san, hoac go truc tiep ten experiment moi)",
            style="Muted.TLabel"
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=6)

        self.progress = ttk.Progressbar(self, mode="indeterminate", length=380)
        self.status_var = tk.StringVar(value="San sang.")
        self.status_label = ttk.Label(self, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.pack(pady=(24, 4))

        self.run_button = ttk.Button(
            self, text="▶  Chay BKERW", width=28, style="Primary.TButton",
            command=self.run_algorithm
        )
        self.run_button.pack(pady=12, ipady=4)

        ttk.Button(self, text="←  Quay lai", style="Ghost.TButton",
                   command=lambda: controller.show_frame(HomePage)).pack(pady=10)

    def on_show(self):
        """Lam moi danh sach experiment moi khi vao trang (phong khi vua tao config moi)."""
        experiments = get_available_experiments()
        self.combo.config(values=experiments)
        if experiments and not self.experiment_var.get():
            self.combo.current(0)

    def run_algorithm(self):
        experiment_name = self.experiment_var.get().strip()
        if not experiment_name:
            messagebox.showwarning("Thieu thong tin", "Vui long chon hoac nhap ten Experiment.")
            return

        if not os.path.isfile(MAIN_SCRIPT):
            messagebox.showerror("Loi", f"Khong tim thay script tai:\n{MAIN_SCRIPT}")
            return

        self.run_button.config(state="disabled")
        self.status_label.config(style="Status.TLabel")
        self.status_var.set(f"Dang chay BKERW voi experiment='{experiment_name}' ...")
        self.progress.pack(pady=(0, 10))
        self.progress.start(12)

        threading.Thread(
            target=self._run_in_background, args=(experiment_name,), daemon=True
        ).start()

    def _run_in_background(self, experiment_name):
        # Lấy param tương ứng với dataset
        params = DATASET_PARAMS.get(experiment_name, DEFAULT_PARAMS)

        cmd = [
            sys.executable,
            MAIN_SCRIPT,
            f"method={BKERW_METHOD_VALUE}",
            f"experiment={experiment_name}",
            f"params.restart_prob={params['restart_prob']}",
            f"params.alpha={params['alpha']}",
            f"params.beta={params['beta']}",
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
            )
            success = result.returncode == 0
            output_log = (result.stdout or "") + "\n" + (result.stderr or "")
        except Exception as e:
            success = False
            output_log = str(e)

        self.after(0, self._on_run_finished, success, output_log, experiment_name)

    def _on_run_finished(self, success, output_log, experiment_name):
        self.run_button.config(state="normal")
        self.progress.stop()
        self.progress.pack_forget()

        if success:
            self.status_label.config(style="Success.TLabel")
            self.status_var.set("Hoan thanh!")
            messagebox.showinfo(
                "Hoan thanh",
                f"Chay thuat toan BKERW cho experiment '{experiment_name}' thanh cong!\n\n"
                "Ket qua (result.txt) va so lieu danh gia (metrics.csv) da duoc luu "
                "trong thu muc output cua Hydra cho lan chay nay."
            )
        else:
            self.status_label.config(style="Error.TLabel")
            self.status_var.set("Co loi xay ra khi chay.")
            messagebox.showerror(
                "Loi khi chay",
                f"Chay thuat toan BKERW cho experiment '{experiment_name}' that bai.\n\n"
                f"Chi tiet loi:\n{output_log[-1500:]}"
            )


if __name__ == "__main__":
    App().mainloop()
import os


class ReportResults():
    def __init__(self, cfg, save_dir='results/results_output/', index_run=0, init_approach=None):
        """
        Initialize the reporter with config and save directory.
        Creates file path: results/results_output/{dataset_name}_{model_name}_{peft_type}.txt
        """
        os.makedirs(save_dir, exist_ok=True)
        if init_approach is  None:
            self.file_path = os.path.join(save_dir, f"{cfg.dataset.name}_{cfg.llm.model_name}_{cfg.peft.type}_{index_run}.txt")
        else:
            self.file_path = os.path.join(save_dir, f"{cfg.dataset.name}_{cfg.llm.model_name}_{cfg.peft.type}_{index_run}_{init_approach}.txt")

    def report_title(self, title):
        """Write a title to the results file."""
        with open(self.file_path, 'a', encoding='utf-8') as f:
            width = 70
            border = '=' * width
            centered_title = title.center(width)
            f.write(f"\n{border}\n")
            f.write(f"{centered_title}\n")
            f.write(f"{border}\n\n")

    def report_txt(self, txt):
        """Write text to the results file."""
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(f"{txt}\n")

    def report(self, title, txt):
        """Report both title and text."""
        self.report_title(title)
        self.report_txt(txt)

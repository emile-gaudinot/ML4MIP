# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python (cake)
#     language: python
#     name: cake
# ---

# %%

import mlflow
import pandas as pd
from IPython.display import Image, display
from ipywidgets import widgets
from mlflow.tracking import MlflowClient

# Adjust the display setting for rows
pd.set_option("display.max_rows", None)

# To display all columns, if needed:
pd.set_option("display.max_columns", None)


# %%
class Dashboard:
    def __init__(self, tracking_uri=None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.experiments = self._fetch_experiments()

        # Widgets
        self.experiment_dropdown = self._create_experiment_dropdown()
        self.runs_output = widgets.Output()
        self.run_dropdown = widgets.Dropdown(description="Run ID:")
        self.metrics_output = widgets.Output()
        self.params_output = widgets.Output()
        self.image_dropdown = widgets.Dropdown(description="Artifact:")
        self.image_output = widgets.Output()

        # Horizontal Box for Metrics and Parameters
        self.details_hbox = widgets.HBox([self.metrics_output, self.params_output])

        # Setup Observers
        self._setup_observers()

    def _fetch_experiments(self):
        """Fetch all experiments and return as a dictionary."""
        experiments = self.client.search_experiments()
        return {exp.name: exp.experiment_id for exp in experiments}

    def _create_experiment_dropdown(self):
        """Create a dropdown widget for selecting experiments."""
        return widgets.Dropdown(
            options=self.experiments,
            description="Experiment:",
            value=None  # Default to None to prompt user selection
        )

    def _setup_observers(self):
        """Set up observers for widget interactions."""
        self.experiment_dropdown.observe(self._on_experiment_change, names="value")
        self.run_dropdown.observe(self._on_run_change, names="value")
        self.image_dropdown.observe(self._on_image_change, names="value")

    def _on_experiment_change(self, change):
        """Callback when the experiment selection changes."""
        if change["type"] == "change" and change["name"] == "value":
            self._display_runs_table(change["new"])

    def _display_runs_table(self, experiment_id):
        """Fetch and display runs for the selected experiment."""
        with self.runs_output:
            self.runs_output.clear_output(wait=True)
            runs = self.client.search_runs(experiment_ids=[experiment_id])
            if not runs:
                print("No runs found for this experiment.")
                return

            # Display runs table
            run_data = []
            for run in runs:
                run_info = {
                    "Run ID": run.info.run_id,
                    "Run Name": run.data.tags.get("mlflow.runName", "Unnamed Run"),
                    "Model Type": run.data.params.get("model.model_type", "Unknown"),
                    "Start Time": pd.to_datetime(run.info.start_time, unit="ms"),
                    "End Time": pd.to_datetime(run.info.end_time, unit="ms") if run.info.end_time else None,
                    "Status": run.info.status
                }
                # Metrics Data:
                # run_info.update({f"Metric: {k}": v for k, v in run.data.metrics.items()})
                run_data.append(run_info)

            runs_df = pd.DataFrame(run_data)
            display(runs_df)

            # Populate run dropdown
            self.run_dropdown.options = [(f"{row['Run Name']} ({row['Run ID']})", row["Run ID"]) for _, row in runs_df.iterrows()]
            self.run_dropdown.value = None

    def _on_run_change(self, change):
        """Callback when the run selection changes."""
        if change["type"] == "change" and change["name"] == "value":
            self._update_run_details(change["new"])

    def _update_run_details(self, run_id):
        """Update metrics and parameters for the selected run."""
        # Fetch the run details
        run = self.client.get_run(run_id)
        metrics = run.data.metrics
        params = run.data.params

        # Clear and update metrics output
        with self.metrics_output:
            self.metrics_output.clear_output(wait=True)
            metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
            display(metrics_df)

        # Clear and update parameters output
        with self.params_output:
            self.params_output.clear_output(wait=True)
            params_df = pd.DataFrame(params.items(), columns=["Parameter", "Value"])
            display(params_df)

        # Populate the image dropdown with PNG artifacts
        artifacts = [artifact.path for artifact in self.client.list_artifacts(run_id)]
        png_artifacts = [artifact for artifact in artifacts if artifact.endswith(".png")]
        self.image_dropdown.options = png_artifacts
        self.image_dropdown.value = None

    def _on_image_change(self, change):
        """Callback when the image selection changes."""
        if change["type"] == "change" and change["name"] == "value":
            self._display_image(change["new"])

    def _display_image(self, artifact_path):
        """Display the selected image artifact."""
        with self.image_output:
            self.image_output.clear_output(wait=True)
            if artifact_path:
                run_id = self.run_dropdown.value
                local_path = self.client.download_artifacts(run_id, artifact_path)
                display(Image(filename=local_path))

    def display(self):
        """Display the full dashboard."""
        display(
            self.experiment_dropdown,
            self.runs_output,
            self.run_dropdown,
            self.details_hbox,  # Render metrics and params in a horizontal box
            self.image_dropdown,
            self.image_output
        )

# TODO: instead of displaying the metrics in a table, add another dropdown
# next to RUN ID that allows to select the metric for the current run



# %%
# dashboard = Dashboard(tracking_uri="/Users/pvmeng/Documents/ML4MIP/runs")
dashboard = Dashboard(tracking_uri="/group/cake/ML4MIP/runs")
dashboard.display()

# %%

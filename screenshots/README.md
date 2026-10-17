# Screenshot checklist

Save PNGs here with these names so the report can reference them.

| File | What to capture |
|---|---|
| 01_mlflow_experiments.png | MLflow UI, experiment list with the three parent runs |
| 02_mlflow_compare.png | MLflow "Compare runs" chart (CV ROC-AUC across models) |
| 03_mlflow_run_artifacts.png | Artifacts tab of the selected run (plots + model) |
| 04_ci_pipeline_success.png | GitHub Actions run graph: lint → test → train → docker all green |
| 05_ci_pipeline_failure.png | A deliberately broken test failing the pipeline, with the log |
| 06_ci_artifacts_summary.png | Job summary metrics table + uploaded artefacts |
| 07_docker_build_run.png | `docker build` and `docker run` output in the terminal |
| 08_docker_predict.png | curl /predict response against the container |
| 09_k8s_pods_svc.png | `kubectl -n heart-risk get pods,svc,hpa` |
| 10_k8s_predict.png | curl /predict via the LoadBalancer / Ingress |
| 11_swagger_docs.png | /docs page of the API |
| 12_grafana_dashboard.png | Grafana dashboard after generating traffic |
| 13_prometheus_targets.png | Prometheus Status → Targets showing each pod UP |
| 14_api_logs.png | `kubectl logs` showing JSON request/prediction logs |
| 15_mlflow_ablation.png | feature_set_ablation.png in the selected run (or MLflow chart grouped by feature_set) |
| 16_calibration_subgroups.png | calibration_comparison.png and subgroup_by_sex.png |
| 17_data_validation_failure.png | CI failing with a clear DataValidationError (e.g. a corrupted CSV on a test branch) |

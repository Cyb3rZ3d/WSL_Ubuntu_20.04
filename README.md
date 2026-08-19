# Amazon Co-Purchase Recommendation System

A CSEC 5311 Big Data Analysis and Security semester project that uses Amazon co-purchasing data, Apache Spark, and Alternating Least Squares (ALS) collaborative filtering to generate product recommendations.

## Submitted Project Scope

The project parses the Amazon product metadata dataset from the Stanford Network Analysis Project (SNAP), converts co-purchase relationships into implicit user-item interactions, trains an ALS model, exports recommendation output, and visualizes the results with Matplotlib, Seaborn, Power BI, and the Spark Web UI.

## Established Project Files

- `example.py` - parses product blocks, trains ALS, and exports recommendations.
- `example-1.py` - documented version of the recommendation workflow.
- `example-2.py` - adds an 80/20 split, RMSE evaluation, a top-products chart, and a user-item heatmap.
- `amazon_recommendations_output.csv` - 100 recorded recommendation rows with user ID, item ID, ASIN, title, group, and predicted rating.
- `top_10_recommendations.png` - generated top-products visualization.
- `user_item_heatmap.png` - generated recommendation heatmap.
- `requirements.txt` - PySpark, pandas, Seaborn, and Matplotlib dependencies.
- `proposed_project_plan.txt` - original project objective and implementation plan.

## Implemented Pipeline

1. Load the semi-structured Amazon metadata in PySpark.
2. Parse ASIN, title, group, and similar-product fields with regular expressions.
3. Map ASIN values to integer identifiers.
4. Represent co-purchase links as implicit interactions with a uniform rating of `1.0`.
5. Split the interaction data into 80% training and 20% testing data.
6. Train a PySpark ALS model with 10 iterations and `regParam=0.1`.
7. Evaluate held-out predictions with RMSE.
8. Generate ten recommendations for ten users and export 100 rows to CSV.
9. Visualize recommendations with Matplotlib, Seaborn, Power BI, and Spark Web UI dashboards.

## Recorded Result

The submitted report records an RMSE of **0.3904** on the held-out test set. This value is reported from the original project execution and has not been newly reproduced for this portfolio update.

## Development Environment

WSL Ubuntu, VS Code, Python 3.10, Apache Spark/PySpark, pandas, Matplotlib, Seaborn, Power BI, and the Spark Web UI at `localhost:4040`.

## Report Evidence

The following 15 original figures were extracted from the submitted semester report and preserved in report order.

| Sequence | Figure |
|---:|---|
| 01 | [Development environment](report-evidence/01-development-environment.png) |
| 02 | [Dataset parsing code](report-evidence/02-dataset-parsing-code.png) |
| 03 | [ALS training code](report-evidence/03-als-training-code.png) |
| 04 | [Visualization code](report-evidence/04-visualization-code.png) |
| 05 | [Model execution](report-evidence/05-model-execution.png) |
| 06 | [Top recommendations](report-evidence/06-top-recommendations.png) |
| 07 | [User-item heatmap](report-evidence/07-user-item-heatmap.png) |
| 08 | [Recommendation output](report-evidence/08-recommendation-output.png) |
| 09 | [Power BI dashboard](report-evidence/09-power-bi-dashboard.png) |
| 10 | [Spark jobs](report-evidence/10-spark-jobs.png) |
| 11 | [Spark stages](report-evidence/11-spark-stages.png) |
| 12 | [Spark storage](report-evidence/12-spark-storage.png) |
| 13 | [Spark environment](report-evidence/13-spark-environment.png) |
| 14 | [Spark executors](report-evidence/14-spark-executors.png) |
| 15 | [Spark SQL](report-evidence/15-spark-sql.png) |

## Project Boundaries

- The source files and recorded outputs remain in their original course-project state.
- The scripts contain the original WSL file paths and require the SNAP `amazon-meta.txt` dataset to run.
- The Power BI dashboard was verified from the submitted `.pbix` artifact; the source dashboard file is not added by this documentation update.
- The RMSE and screenshots are preserved from the submitted work rather than regenerated.

## Portfolio

[View the digital project profile](https://cyb3rz3d.github.io/amazon-co-purchase-recommendation.html)

The open-source project you mentioned, `IBUMBLEBEE/grafana-alert4ml-datasource`, is a vivid and highly representative example. It does not contradict the earlier points — on the contrary, it **exactly corroborates the "funnel-shaped architecture" and how industry handles complex algorithms**.

The idea this project represents is, at its core, about how complex algorithms can "land in an existing monitoring ecosystem". Looking at its design positioning shows its relationship to built-in base algorithms such as those in GreptimeDB:

### 1. It sits at the "second layer of the architecture (the bottom of the funnel)"

In the funnel-shaped architecture described earlier:

* **Layer 1 (underlying database):** databases like GreptimeDB rely on built-in `zscore`/`mad` for mass-data pre-filtering and stream processing at the storage layer.
* **Layer 2 (visualization and advanced analysis):** **`grafana-alert4ml-datasource` sits exactly in this layer.** Its core idea is to treat Grafana as a "data relay (Meta-Datasource)": pull already-filtered data from existing upstream data sources such as Prometheus and Elasticsearch, apply machine learning (ML) anomaly detection inside this custom datasource plugin, and finally render the results in Grafana.

### 2. It solves "low-frequency analysis", not "storage and high-frequency ingestion"

If you run complex algorithms in the underlying database (for example, stuffing a complicated graph neural network into GreptimeDB), the database would peg its CPU and crash under millions of writes per second.

The smart part of this Grafana plugin is: **it moves the computation to the Grafana side (or its backend proxy layer)**.

* The underlying database still only handles fast storage and retrieval.
* The plugin only requests data and runs the ML algorithm when an operator opens a particular dashboard, or a low-frequency alert rule fires (for example, every 5 minutes).
* This architecture relieves the database of computation pressure, giving complex ML algorithms room to breathe and compute.

### 3. The typical industry pain points this project faces (why it is at POC stage)

Based on the developer's discussions in the community (e.g. Reddit), this project is currently more of a POC (proof of concept). It is cool, but it also exposes the downsides of complex algorithms in real-world deployment quite clearly:

* **Latency from the long data path:** the normal flow is `data source -> Grafana display`.
  With this plugin the flow is: `upstream data source -> converted into a DataFrame by Grafana -> fed to your ML plugin to run the model -> converted back to Grafana for rendering`.
  This adds extra network and serialization overhead, which is too much latency for high-frequency, real-time "second-level alerts".
* **The high-cardinality disaster still exists:**
  Although it runs on Grafana, if the user's system has tens of thousands of Pod nodes and each node has to run ML detection through this plugin, the Grafana service's own memory and CPU will quickly be drained by this plugin.
* **Cold start and model maintenance cost:**
  Complex algorithms (especially ML) usually need history data (e.g. the past 7 days to recognize periodicity). Every Grafana request has to pull the past 7 days of data on the fly to "train" or "infer", putting enormous query pressure (Query Blast) on the underlying Prometheus/GreptimeDB.

### Summary: how the two complement each other

* **GreptimeDB's `zscore / mad / iqr`** solves the **"breadth"** problem: flood irrigation — raising the first line of defense for thousands of metrics at extremely low cost and extremely high speed.
* **Projects like `grafana-alert4ml-datasource`** solve the **"depth"** problem: for a few core metrics (e.g. the transaction success rate of the core business, the QPS of the core dashboard), after the first line of defense filters, or at visualization time, provide smarter ML anomaly identification.

So in real production environments you will often see this combination: **the bottom layer uses GreptimeDB/Prometheus with basic statistical algorithms for 90% of routine fast alerts; the top layer uses ML plugins or AI-Ops platforms like this one to chew on the 10% most difficult, most period-sensitive golden metrics.**

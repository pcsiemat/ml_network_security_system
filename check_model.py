import dagshub
import mlflow

dagshub.init(repo_owner="pcsiemat", repo_name="ml_network_security_system", mlflow=True)

model = mlflow.sklearn.load_model("models:/Random_Forest/2")
print(model)
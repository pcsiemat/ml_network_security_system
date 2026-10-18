#def main():
#    print("Hello from ml-end-network-security-system!")




from ml_network_security_system.components.data_ingestion import DataIngestion
from ml_network_security_system.exception.exception import AppException
from ml_network_security_system.logger.log import logging
from ml_network_security_system.entity.config_entity import DataIngestionConfig
from ml_network_security_system.entity.config_entity import TrainingPipelineConfig

import sys
if __name__ == "__main__":
    try:
        traningpipelineconfig = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(traningpipelineconfig)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiate the data ingestion")
        dataingestionartifac =  data_ingestion.initiate_data_ingestion()
        print(dataingestionartifac)


    except Exception as e:
        raise AppException(e,sys)
    #main()

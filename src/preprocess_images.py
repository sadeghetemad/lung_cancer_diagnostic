import boto3
from botocore.config import Config
import sagemaker
import os
import time
from sagemaker.processing import ScriptProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
import time
from time import gmtime, strftime
from botocore.exceptions import ClientError


SOLUTION_BUCKET = 'nsclc-medical-image-data-811165582441-eu-west-2-an'
SOLUTION_PREFIX = 'ct-seg-image'
SOLUTION_NAME = 'lung-cancer-diagnostic'
BUCKET = 'nsclc-medical-image-data-811165582441-eu-west-2-an'

REGION = 'eu-west-2'
ECR_REPOSITORY = '811165582441.dkr.ecr.eu-west-2.amazonaws.com/medical-image-processing'
ECR_IMAGE_URI = '811165582441.dkr.ecr.eu-west-2.amazonaws.com/medical-image-processing:0.0.1-staging'
ACCOUNT_ID = 811165582441
ROLE = 'arn:aws:iam::811165582441:role/lung_cancer_diagnostic'

SM_BOTO = boto3.client('sagemaker', config=Config(connect_timeout=5, read_timeout=60, retries={'max_attempts': 20}), region_name=REGION)
sagemaker_session = sagemaker.Session(sagemaker_client=SM_BOTO)



def launch_processing_job(subject, input_data_s3, output_data_s3, feature_group_name, offline_store_s3uri, retries):
    
    exp_datetime = strftime('%Y-%m-%d-%H-%M-%S', gmtime())
    jobname = f'{SOLUTION_PREFIX}-{subject}-{exp_datetime}' 

    inputs = [ProcessingInput(input_name='DICOM',
                              source=f'{input_data_s3}/{subject}', 
                              destination='/opt/ml/processing/input')]

    outputs = [ProcessingOutput(output_name=i,
                                source='/opt/ml/processing/output/%s' % i,
                                destination=f"{output_data_s3}/{i}") 
               for i in ['CT-Nifti', 'CT-SEG', 'PNG']]

    arguments = ['--subject', subject, 
                 '--feature_group_name', feature_group_name, 
                 '--offline_store_s3uri', offline_store_s3uri]

    script_processor = ScriptProcessor(command=['python3'],
                                       image_uri=ECR_IMAGE_URI,
                                       role=ROLE,
                                       instance_count=1,
                                       instance_type='ml.t3.medium',
                                       volume_size_in_gb=5,
                                       sagemaker_session=sagemaker_session,
                                       env={
                                                "SAGEMAKER_ROLE_ARN": ROLE
                                            })
    current_retry = 1
    while True and current_retry <= retries:
        try:
            script_processor.run(code='../src/dcm2nifti_processing.py',
                                 inputs=inputs,
                                 outputs=outputs,
                                 arguments=arguments,
                                 job_name=jobname,
                                 wait=False,
                                 logs=False)
            return script_processor

        except ClientError as e:
            if "No S3 objects found under S3 URL" in str(e):
                print("Bad input data has been removed from s3, processing is not created!")
                return None
            elif "ResourceLimitExceeded" in str(e):
                if current_retry == retries:
                    raise
                else:
                    print("Resource reaches limit, please retry after 30 seconds ...")
                    current_retry += 1
                    time.sleep(30)
                    continue
            else:
                print(f'Processing job with {subject} subject is not created successfully! {e}.')
                raise

def query_jobs(dict_processor):
    for key in list(dict_processor):
        if dict_processor[key]:
            status = dict_processor[key].jobs[-1].describe()['ProcessingJobStatus']
            if status in ["Completed", "Failed", "Stopped"]:
                del dict_processor[key]
        else:
            del dict_processor[key] # when no ProcessingJob created, i.e., None
    return len(dict_processor)


def wait_for_instance_quota(dict_processor, job_limit=4, wait=30):   
    job_count = query_jobs(dict_processor)    
    while job_count >= job_limit:
        print(f'Current total running jobs {job_count} is reaching the limit {job_limit}. Waiting {wait} seconds...')
        time.sleep(wait)
        job_count = query_jobs(dict_processor)
        
    print(f'Current total running jobs {job_count} is below {job_limit}. Proceeding...')
    return


if __name__ == '__main__':
    
    dict_processor = {}

    input_data_bucket = SOLUTION_BUCKET
    input_data_prefix = f"nsclc_radiogenomics"

    output_data_bucket=BUCKET
    output_data_prefix= "processed/nsclc_radiogenomics"

    input_dicom_dir = f"s3://{input_data_bucket}/{input_data_prefix}"
    output_nifti_dir = f"s3://{output_data_bucket}/{output_data_prefix}"

    imaging_feature_group_name = f'{SOLUTION_PREFIX}-imaging-feature-group'

    offline_store_s3uri = '%s/multimodal-imaging-featurestore' % output_nifti_dir

    subject_list = ['R01-%03d'%i for i in range(1, 164)]

    for subject in subject_list:
        print(subject)
        wait_for_instance_quota(dict_processor, job_limit=10, wait=180)
        dict_processor[subject] = launch_processing_job(subject, input_dicom_dir, output_nifti_dir, imaging_feature_group_name, offline_store_s3uri, 10)
        time.sleep(5)
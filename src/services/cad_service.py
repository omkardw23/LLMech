from kittycad.client import ClientFromEnv
from kittycad.api.ml import create_text_to_cad, get_text_to_cad_model_for_user
from kittycad.models import (
    ApiCallStatus,
    Error,
    FileExportFormat,
    TextToCad,
    TextToCadCreateBody,
)
import time
from utils.logger import logger

class CADService:
    def __init__(self):
        self.client = ClientFromEnv()

    def generate_3d_model(self, prompt):
        try:
            response = create_text_to_cad.sync(
                client=self.client,
                output_format=FileExportFormat.GLTF,
                body=TextToCadCreateBody(prompt=prompt),
            )

            if isinstance(response, Error) or response is None:
                logger.error(f"Error generating 3D model: {response}")
                return None

            result = self._poll_until_complete(response)
            return self._save_model(result) if result else None

        except Exception as e:
            logger.error(f"Error in generate_3d_model: {str(e)}")
            return None

    def _poll_until_complete(self, result: TextToCad):
        while result.completed_at is None:
            time.sleep(5)
            response = get_text_to_cad_model_for_user.sync(
                client=self.client,
                id=result.id,
            )
            if isinstance(response, Error) or response is None:
                return None
            result = response

        return result if result.status == ApiCallStatus.COMPLETED else None

    def _save_model(self, result):
        if not result.outputs:
            return None
            
        output_path = "generated_model.gltf"
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(result.outputs["source.gltf"].decode("utf-8"))
        return output_path 
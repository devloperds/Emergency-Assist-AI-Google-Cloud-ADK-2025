# main_agent.py
import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import logging

# Import our tools
from tools import (
    get_user_location,
    analyze_traffic_and_hospital_availability,
    send_alert_to_hospital,
    notify_emergency_contacts,
    USER_INFO
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize Gemini API
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    raise


def run_emergency_protocol():
    """
    Orchestrates the entire emergency response using the Gemini agent.
    """
    try:
        # 1. Initialize the Gemini Model with the tools
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            tools=[
                get_user_location,
                analyze_traffic_and_hospital_availability,
                send_alert_to_hospital,
                notify_emergency_contacts
            ]
        )

        chat = model.start_chat(enable_automatic_function_calling=True)

        # 2. Define the Prompt / The Plan for the AI
        prompt = """
        You are an AI emergency response agent. A user needs to go to a hospital urgently.
        Your task is to automatically contact the best possible hospital and notify contacts.
        Follow these steps strictly in order:

        1. First, get the user's current location.
        2. Once you have the location, analyze the traffic and availability for all pre-selected hospitals.
        3. From the analysis results, you MUST decide on the single best hospital. 
           The best hospital is one that is 'Accepting' patients and has the lowest travel time. 
           Ignore hospitals that are 'Diverting'.
        4. After selecting the best hospital, send an alert to THAT hospital with all the user's details.
        5. Finally, after the hospital has been alerted, notify the user's emergency contacts.
        6. Confirm when the process is complete. Do not ask me for more information, just execute the plan.

        User medical information:
        - Name: {name}
        - Age: {age}
        - Blood Type: {blood_type}
        - Known Conditions: {conditions}
        """.format(
            name=USER_INFO.get("name", "Unknown"),
            age=USER_INFO.get("age", "Unknown"),
            blood_type=USER_INFO.get("blood_type", "Unknown"),
            conditions=", ".join(USER_INFO.get("conditions", [])) or "None"
        )

        print("--- STARTING EMERGENCY PROTOCOL ---")
        print(f"USER: Help, I need to go to the hospital!")

        # 3. Send the prompt to the model and let it run
        try:
            response = chat.send_message(prompt)

            print("\n--- EMERGENCY PROTOCOL COMPLETE ---")
            final_text_response = ''.join(part.text for part in response.parts)
            print(f"AGENT's FINAL CONFIRMATION: {final_text_response}")

            return {
                "status": "success",
                "message": final_text_response
            }

        except Exception as e:
            logger.error(f"Error during protocol execution: {e}")
            return {
                "status": "error",
                "message": f"Protocol failed: {str(e)}"
            }

    except Exception as e:
        logger.error(f"Failed to initialize emergency protocol: {e}")
        raise


if __name__ == "__main__":
    try:
        result = run_emergency_protocol()
        if result["status"] == "error":
            print(f"\nERROR: {result['message']}")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {str(e)}")
        print("Please check your configuration and try again.")
import os
import time
import logging
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)

def get_llm(temperature=0.4, response_mime_type=None, safety_settings=None, model=None):
    """
    Creates a ChatGoogleGenerativeAI instance with configured safety settings and default retries.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        google_api_key = os.getenv("GEMINI_API_KEY")
        
    if safety_settings is None:
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        
    selected_model = model or os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
    
    kwargs = {
        "model": selected_model,
        "temperature": temperature,
        "api_key": google_api_key,
        "safety_settings": safety_settings,
        "max_retries": 3,
    }
    if response_mime_type:
        kwargs["response_mime_type"] = response_mime_type
        
    return ChatGoogleGenerativeAI(**kwargs)

def safe_llm_invoke(primary_llm, messages, max_retries=3, initial_delay=2.0, fallback_llm=None):
    """
    Wrapper around llm.invoke() that catches transient API exceptions and retries with exponential backoff.
    If all retries for the primary model fail and a fallback model is provided, it tries invoking the fallback.
    """
    delay = initial_delay
    last_exc = None
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Invoking LLM (attempt {attempt}/{max_retries}) model: {getattr(primary_llm, 'model', 'unknown')}")
            return primary_llm.invoke(messages)
        except Exception as e:
            last_exc = e
            err_str = str(e)
            class_name = e.__class__.__name__
            module_name = e.__class__.__module__
            
            logger.warning(f"LLM invocation attempt {attempt} failed: {class_name} ({module_name}): {err_str}")
            
            # Identify transient/retryable failures (503 UNAVAILABLE, 429 rate limit, 500 internal errors)
            is_transient = False
            # Check string content for common error indicators
            err_lower = err_str.lower()
            if any(term in err_lower for term in ["503", "502", "504", "429", "500", "unavailable", "rate limit", "resourceexhausted", "high demand"]):
                is_transient = True
            # Check specific class or module names
            if class_name in ["ServerError", "APIError", "GoogleAPICallError", "InternalServerError", "ServiceUnavailable", "ResourceExhausted"]:
                is_transient = True
            if "google.genai" in module_name or "google.api_core" in module_name:
                is_transient = True
                
            if attempt == max_retries:
                logger.error(f"LLM primary invocation failed all {max_retries} attempts.")
                break
                
            if is_transient:
                logger.warning(f"Transient error identified. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2.0
            else:
                logger.error(f"Non-transient error identified. Raising exception immediately.")
                raise e
                
    # If primary model failed all retries, try the fallback model if available
    if fallback_llm is not None:
        logger.warning(f"Primary model failed all attempts. Trying fallback LLM: {getattr(fallback_llm, 'model', 'unknown')}")
        try:
            return fallback_llm.invoke(messages)
        except Exception as fallback_err:
            logger.error(f"Fallback LLM invocation also failed: {fallback_err}", exc_info=True)
            raise fallback_err
            
    if last_exc:
        raise last_exc
    raise RuntimeError("LLM invocation failed with unknown error")

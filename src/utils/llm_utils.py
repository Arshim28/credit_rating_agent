from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

def init_chat_model(model: str, model_provider: str = None) -> BaseChatModel:
    """
    Initialize a chat model based on the model name and provider.
    
    Args:
        model: The model name to use
        model_provider: Optional provider name to override auto-detection
        
    Returns:
        BaseChatModel: The initialized chat model
    """
    if model_provider is None:
        # Auto-detect provider from model name
        if model.startswith("claude"):
            model_provider = "anthropic"
        elif model.startswith("gpt") or model.startswith("text-davinci") or model.startswith("o3"):
            model_provider = "openai"
        elif model.startswith("mistral") or model.startswith("mixtral"):
            model_provider = "mistralai"
        elif model.startswith("gemini"):
            model_provider = "google"
        elif model.startswith("llama"):
            model_provider = "ollama"
        else:
            # Default to OpenAI if we can't determine the provider
            model_provider = "openai"

    # Import the appropriate model based on provider
    if model_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model_name=model)
    elif model_provider == "openai":
        from langchain_openai import ChatOpenAI
        # Handle 'openai:' prefix if given directly
        if model.startswith("openai:"):
            model = model[7:]
        return ChatOpenAI(model_name=model)
    elif model_provider == "mistralai":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model_name=model)
    elif model_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model)
    elif model_provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model)
    elif model_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model_name=model)
    elif model_provider == "cohere":
        from langchain_cohere import ChatCohere
        return ChatCohere(model=model)
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")
from foundry_local_sdk import FoundryLocalManager, Configuration

m = FoundryLocalManager(Configuration(app_name="test"))
m.start_web_service()

variant = m.catalog.get_model_variant("Phi-3.5-mini-instruct-generic-cpu:2")
print("Indiriliyor...")
variant.download()
print("Tamamlandi!")

cached = m.catalog.get_cached_models()
print("Cache:", cached)

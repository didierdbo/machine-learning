import torch
import time

# Vérification de base
print(f"Version de PyTorch : {torch.__version__}")
print(f"CUDA disponible : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # Création de deux grosses matrices aléatoires sur le GPU
    print(f"Calcul en cours sur : {torch.cuda.get_device_name(0)}...")
    
    start_time = time.time()
    
    # On crée des tenseurs (matrices) de 10000x10000 sur le GPU
    x = torch.randn(10000, 10000).to("cuda")
    y = torch.randn(10000, 10000).to("cuda")
    
    # Multiplication de matrices (opération de base du ML)
    z = torch.matmul(x, y)
    
    # On attend que le GPU finisse pour mesurer le temps
    torch.cuda.synchronize()
    
    end_time = time.time()
    print(f"✅ Calcul terminé avec succès en {end_time - start_time:.4f} secondes !")
else:
    print("❌ Le GPU n'est pas utilisé. Vérifie ton installation.")
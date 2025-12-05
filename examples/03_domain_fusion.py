from pathlib import Path

from unidomain import domain_fusion_pipeline

root_dir = Path(__file__).parent.parent.resolve()


domain_dir = root_dir / "outputs" / "02_atomic_domain"
if not domain_dir.exists():
    raise FileNotFoundError(
        f"Tutorial-Domain Fusion: {domain_dir} does not exist. Please run 'examples/02_atomic_domain.py' first."
    )
    
save_dir = root_dir / "outputs" / "03_domain_fusion"
domain_fusion_pipeline(domain_dir, save_dir, num_workers=2)



# configs = {
#     10: 4,
#     12: 4,
#     14: 4,
#     16: 6,
#     18: 6,
#     20: 6
# }
# def output_command():
#     domain_dir = root_dir / "data" / "sampled_domains"
#     output_dir = root_dir / "examples" / "domain_fusion"
#     for sample, num_workers in configs.items():
#         sub_domain = domain_dir / f"sample_{sample}"
#         sub_output = output_dir / f"sample_{sample}"
#         print(f"unidomain fusion -i {sub_domain} -o {sub_output} -n {num_workers}")

# output_command()
# domain_fusion_pipeline(domain_dir, output_dir, num_workers=2)

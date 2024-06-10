from to_neo4j import parse_vault

if __name__ == "__main__":
    print("Quacks like a duck, looks like a goose.")

    # vault_path = "/home/xoph/SlipBoxCopy/Slip Box"
    vault_path = "/home/xoph/repos/github/nfroseth/world_graph/test_vault"
    notes = parse_vault(vault_path)
    
    print(notes["Good Faith Schedule"].tags)
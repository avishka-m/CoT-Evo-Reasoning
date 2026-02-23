from data_loader import load_dataset, get_all_chains_for_record

d = load_dataset("data/medical_qa.json")
print(f"Records loaded: {len(d)}")

for rec in d:
    chains = get_all_chains_for_record(rec)
    print(f"\nID: {rec['id']} — {len(chains)} chains")
    for c in chains:
        print(f"  [{c['model']}] {len(c['steps'])} steps | preview: {c['text'][:60]}...")

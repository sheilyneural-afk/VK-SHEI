fn main() {
    let adapters = kvslim::adapter_catalog::host_capabilities();
    println!(
        "{}",
        serde_json::to_string_pretty(&adapters).expect("serialize adapters")
    );
}

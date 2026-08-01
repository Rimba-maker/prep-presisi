from prep_presisi.entities import InsightContext


def build_prompt(context: InsightContext) -> str:
    effects = []
    if context.is_weekend:
        effects.append("akhir pekan")
    if context.is_payday_week:
        effects.append("minggu gajian")
    if context.is_ramadan:
        effects.append("Ramadan")
    if context.is_lebaran_week:
        effects.append("minggu Lebaran")
    effect_text = ", ".join(effects) if effects else "hari biasa"

    return (
        "Kamu asisten operasional restoran. Buat SATU kalimat rekomendasi bahasa "
        "Indonesia yang actionable buat operator outlet (bukan data scientist), "
        "berdasarkan data berikut:\n"
        f"- Menu: {context.menu_item_name} di outlet {context.outlet_name}\n"
        f"- Rekomendasi qty prep besok: {context.predicted_qty}\n"
        f"- Rata-rata historis: {context.historical_avg_qty:.0f}\n"
        f"- Kondisi: {effect_text}\n"
        f"- Estimasi waste avoided: Rp {context.waste_avoided_rupiah:,.0f}\n\n"
        "Kalimat harus singkat, langsung actionable (jumlah + alasan singkat), tanpa "
        "pembukaan/basa-basi. Contoh gaya: 'Besok siapkan 42 porsi Soto Ayam — turun "
        "13% dari rata-rata karena bukan minggu gajian & hari biasa.'"
    )

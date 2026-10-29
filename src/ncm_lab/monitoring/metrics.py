"""Detached component/count mapping from the authoritative trainer objective."""

from ncm_lab.monitoring.records import Measure, mean_measure


def quantities(metrics, detail, *, generator, auxiliary_weight, answer_count, stage, split):
    prefix = f"{stage}/{split}/"
    result = {}

    def put(name, value, count=1, reduction="equally weighted microbatch objective", unit="dimensionless"):
        result[prefix + name] = Measure(value, count, unit, reduction)

    put("loss/total_objective", metrics["loss"])
    if generator:
        put("loss/text_ce", metrics["text_loss"])
        put("loss/normalized_auxiliary", metrics["latent_loss"])
        put("loss/weighted_auxiliary", auxiliary_weight * metrics["latent_loss"])
        put("coefficients/auxiliary", auxiliary_weight)
        result[prefix + "data/text_ce"] = mean_measure(
            metrics["text_loss"] * answer_count,
            answer_count,
            unit="nats per supervised token",
            reduction="summed NLL / valid token count; slot t predicts token t",
        )
        active = sum(row["vectors"] > 0 for row in detail["groups"])
        for i, row in enumerate(detail["groups"], 1):
            group = prefix + f"aux/group_{i:02d}/"
            for name, numerator in (("raw_mse", row["raw_sum"]), ("normalized_mse", row["normalized_sum"])):
                result[group + name] = mean_measure(
                    numerator, row["elements"], reduction="squared error sum / valid scalar count"
                )
                result[group + name.replace("mse", "squared_error_sum")] = Measure(
                    numerator,
                    row["elements"],
                    numerator=numerator,
                    reduction="sum; count retained separately",
                )
            result[group + "valid_vectors"] = Measure(
                row["vectors"], unit="vectors", numerator=row["vectors"], reduction="sum; valid vectors"
            )
            result[group + "valid_scalars"] = Measure(
                row["elements"],
                unit="scalar elements",
                numerator=row["elements"],
                reduction="sum; valid scalar elements",
            )
            coefficient = auxiliary_weight / active if row["vectors"] else 0.0
            result[group + "coefficient"] = Measure(coefficient)
            result[group + "weighted_contribution"] = Measure(
                coefficient * (row["normalized_sum"] / row["elements"]) if row["elements"] else 0.0,
                reduction="actual auxiliary coefficient / active groups times cumulative normalized MSE",
            )
    else:
        count = detail["elements"]
        for name, value in (("full", metrics["mse_full"]), ("sampled_cutoff", metrics["mse_cutoff"])):
            put("loss/" + name + "_normalized_mse", value)
            put("loss/" + name + "_weighted", 0.5 * value)
            result[prefix + "data/" + name + "_raw_mse"] = mean_measure(
                value * detail["normalization"] * count,
                count,
                reduction="raw squared error / valid scalar count",
            )
            result[prefix + "data/" + name + "_normalized_mse"] = mean_measure(
                value * count, count, reduction="raw MSE / frozen training variance"
            )
        put("loss/head_kl", metrics["head_kl"])
        put("loss/head_kl_weighted", detail["beta"] * metrics["head_kl"])
        put("coefficients/head_kl", detail["beta"])
        put("coefficients/full_reconstruction", 0.5)
        put("coefficients/cutoff_reconstruction", 0.5)
        result[prefix + "data/head_kl"] = mean_measure(
            metrics["head_kl"] * metrics["head_positions"],
            metrics["head_positions"],
            reduction="sampled cutoff KL(teacher || reconstruction); temperature 1; sampled positions only",
        )
        result[prefix + f"cutoff_{metrics['cutoff'] + 1:02d}/reconstruction_normalized_mse"] = mean_measure(
            metrics["mse_cutoff"] * count, count, reduction="observed cutoff only; sum / scalar count"
        )
        for representation, values in detail["statistics"].items():
            for name in ("mean", "rms", "sampled_pooled_scalar_variance"):
                result[prefix + "statistics/" + representation + "/" + name] = Measure(
                    values[name], values["count"], reduction=values["sampling"]
                )
    return result

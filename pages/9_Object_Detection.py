"""
9_Object_Detection.py
---------------------
Object Detection lesson page (YOLO, results only).

Surfaces the sibling ``object-detection/`` repo: a pretrained YOLO model finding
everyday things in photos, and the same model fine-tuned on ~50 hand-labelled
photos to find something it had no word for — spaghetti bolognese.

**Results only.** The app never runs YOLO. The sibling's ``export_results.py``
stores every box both models found (down to a low confidence) in
``object-detection/results/``; ``utils/detection_utils`` reads that and redraws
the boxes with Pillow, so the confidence slider is still live. That keeps torch
(~2 GB) out of the app — and keeps the app clear of the AGPL licence that
Ultralytics YOLO and models trained with it carry.

Graceful behaviour: the sibling is git-ignored, so the page explains itself if
``object-detection/results/`` is not there.
"""

import re
from pathlib import Path

import plotly.express as px
import streamlit as st

from utils import detection_utils as D
from utils.teaching import source_of

st.set_page_config(page_title="Object Detection", layout="wide", page_icon="🍝")

ROOT = Path(__file__).resolve().parents[1]
SIBLING = ROOT / "object-detection"


def show_source(obj, label: str) -> None:
    """Reveal the real function that runs, so code can't drift from behaviour."""
    with st.expander(label):
        st.code(source_of(obj), language="python")


def photo_name(key: str) -> str:
    return Path(key).stem.replace("_", " ")


def summary_table(frame) -> None:
    """What was found: one row per label, confidence as a bar."""
    st.dataframe(
        D.label_summary(frame),
        hide_index=True,
        width="stretch",
        column_config={"best confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")},
    )


def default_index(options: list[str], stem: str) -> int:
    return next((i for i, key in enumerate(options) if stem in key), 0)


results = D.load_results()

st.title("🍝 Object Detection — teach a computer to see something new")
if results is None:
    st.error(
        "This lesson reads the exported results of the sibling `object-detection/` repo, and "
        f"`{D.RESULTS}` was not found. Clone it next to the app (`make siblings`), or run "
        "`uv run python export_results.py` inside it."
    )
    st.stop()

training = results["training"]
pretrained = D.detections_frame(results, "pretrained")
fine_tuned = D.detections_frame(results, "fine_tuned")

st.markdown(
    f"""
    A phone camera that finds faces, a self-checkout that recognises bananas, a car that brakes
    for pedestrians — all of them run an **object detector**: a model that looks at a photo and
    answers *what is where*. This lesson shows that building one for **your own** problem is
    far less work than it sounds: **{training['train_images'] + training['val_images']} labelled
    photos** and **{training['training_minutes']:.0f} minutes on a laptop** taught a free model to
    recognise a new kind of object.
    """
)
with st.container(border=True):
    st.markdown(
        """
        💼 **Where you'll use this** — counting products on a shelf to spot gaps, checking parcels
        or parts for damage at the end of a line, counting visitors or cars from a camera, or
        sorting thousands of insurance-claim photos by what is in them. Whenever someone says
        *"we'd need a person to look at every picture"*, a detector is worth a try.
        """
    )
st.info(
    "All photos on this page are openly licensed (credits in the last tab). The models ran "
    "in the sibling repo; this page shows their stored results, so it needs no special hardware."
)

tab_see, tab_new, tab_learn, tab_compare, tab_code, tab_licence = st.tabs(
    [
        "👀 What a detector sees",
        "🏷️ Teaching it something new",
        "📈 Watching it learn",
        "🔍 Before vs. after",
        "🧑‍💻 All the code it took",
        "📜 Licences & credits",
    ]
)

# ===========================================================================
# 1. A pretrained detector on everyday photos
# ===========================================================================
with tab_see:
    st.subheader("Box, label, confidence — that's the whole answer")
    st.markdown(
        """
        This is **YOLO**, a free detector that was trained on a big public photo collection
        (COCO) and knows **80 everyday things**: people, cars, cups, forks, bananas, dogs …
        For every object it finds, it returns three things:

        1. a **box** — where the object is,
        2. a **label** — what it thinks it is,
        3. a **confidence** — how sure it is, from 0 to 100 %.

        The slider decides how sure the model has to be before we show a box.
        """
    )
    everyday = D.images_for(results, "pretrained", "everyday")
    left, right = st.columns([1, 2])
    with left:
        key = st.selectbox(
            "Photo", everyday, index=default_index(everyday, "dinner_table"),
            format_func=photo_name, key="see_photo",
        )
        threshold = st.slider(
            "Minimum confidence", 0.05, 0.95, results["draw_confidence"], 0.05, key="see_threshold"
        )
        shown = D.above(pretrained[pretrained["image"] == key], threshold)
        st.metric("Boxes shown", len(shown))
        summary_table(shown)
        st.caption(
            "Try **cutlery**: the wooden fork comes out as a *knife*. The model learned from "
            "photos of ordinary metal cutlery — it is best on things that look like what it has seen."
        )
    with right:
        st.image(D.draw_boxes(D.image_path(key), shown), width="stretch")

    st.markdown("##### The one knob you really turn")
    all_boxes = pretrained[pretrained["image"] == key]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(D.threshold_table(all_boxes), hide_index=True, width="stretch")
    with c2:
        st.markdown(
            """
            **Low threshold:** you find almost everything — plus guesses that are wrong.
            **High threshold:** only sure things, but you miss some real objects.

            There is no right value in general. A shop counting stock may accept a few wrong
            boxes; a system that raises a safety alarm may not. *Choosing the threshold is a
            business decision, not a technical one.*
            """
        )
    show_source(D.draw_boxes, "🔍 Show the code that draws the boxes")

# ===========================================================================
# 2. Something it has no word for — and how labelling works
# ===========================================================================
with tab_new:
    st.subheader("Ask it about spaghetti bolognese")
    spaghetti = D.images_for(results, "pretrained", "spaghetti")
    key = st.selectbox(
        "Photo", spaghetti, index=default_index(spaghetti, "spaghetti_brighton"),
        format_func=photo_name, key="new_photo",
    )
    shown = D.above(pretrained[pretrained["image"] == key], results["draw_confidence"])
    c1, c2 = st.columns([2, 1])
    with c1:
        st.image(D.draw_boxes(D.image_path(key), shown), width="stretch")
    with c2:
        st.markdown(
            """
            The pretrained model is not wrong — it simply has **no word for spaghetti
            bolognese**. "Spaghetti" is not one of its 80 things, so the best it can do is a
            *bowl*, a *dining table* or a *fork*.

            That is the situation in most companies: your product, your defect, your shelf
            layout is **not** in any public model. So we teach it.
            """
        )
        summary_table(shown)

    st.markdown("##### The recipe")
    st.graphviz_chart(D.workflow_dot(training))
    st.markdown(
        f"""
        1. **Collect photos** of the new thing — here {training['train_images']} for learning and
           {training['val_images']} kept aside to check the result.
        2. **Label them** in [Label Studio](https://labelstud.io) — a free tool you open in the
           browser, where you draw one box per object and pick its name. For 50 photos this takes
           well under an hour.
        3. **Fine-tune**: start from the pretrained model and let it practise on your photos. It
           keeps everything it already knows about edges, shapes and textures — it only learns
           the new name. That is why so few photos are enough.
        """
    )
    with st.expander("🏷️ What a label actually is — one line of text per box"):
        example = D.detections_frame(results, "fine_tuned").sort_values("confidence").iloc[-1]
        size = results["models"]["fine_tuned"]["images"][example["image"]]
        line = D.to_yolo_label(
            [example["x1"], example["y1"], example["x2"], example["y2"]], size["width"], size["height"]
        )
        st.markdown(
            "When you export from Label Studio in *YOLO format*, every photo gets a small text "
            "file with one line per box: **class number, box centre (x, y), box width and "
            "height** — all as fractions of the photo, so the label still fits if the photo is "
            f"resized. For the box on *{photo_name(example['image'])}* that line is:"
        )
        st.code(line, language="text")
        st.markdown("And the project tells the trainer where the photos are and what the class numbers mean:")
        st.code(
            "train: images/train\nval: images/val\nnc: 1                    # number of classes\n"
            'names: ["spaghetti_bolo"]  # class 0',
            language="yaml",
        )
        show_source(D.to_yolo_label, "🔍 Show the conversion")
    with st.expander("🐳 Starting Label Studio (from the sibling README)"):
        st.code(
            "docker run -it -p 8080:8080 -v $(pwd)/mydata:/label-studio/data "
            "heartexlabs/label-studio:latest",
            language="bash",
        )
        st.caption("Then open http://localhost:8080, create a project, import photos, draw boxes, export as YOLO.")

# ===========================================================================
# 3. Training curves in plain words
# ===========================================================================
with tab_learn:
    st.subheader("Fifty rounds of practice")
    metrics = D.training_metrics()
    best = training["best"]
    reached = D.epochs_to_reach(metrics, "mAP50", 0.9)
    st.markdown(
        f"""
        Training goes in **rounds** (*epochs*): the model looks at all
        {training['train_images']} training photos, adjusts itself a little, and is then tested
        on the {training['val_images']} photos it never learns from. Here it ran
        {training['epochs']} rounds on an ordinary laptop processor — no graphics card — in about
        **{training['training_minutes']:.0f} minutes**.

        YOLO does not simply keep the last round: it keeps the round that scored **best on the
        check photos** — here round **{training['best_epoch']}**. That kept model is the one used
        on every photo on this page, and these are its scores:
        """
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{best['precision']:.0%}", help="Of the boxes it drew, how many were right")
    c2.metric("Recall", f"{best['recall']:.0%}", help="Of the real plates, how many it found")
    c3.metric("mAP50", f"{best['mAP50']:.2f}", help="Overall score with a loose box fit (0 to 1)")
    c4.metric("mAP50-95", f"{best['mAP50-95']:.2f}", help="Overall score with a strict box fit (0 to 1)")
    st.markdown(
        f"""
        - **Precision** — of the boxes it drew, how many were really spaghetti.
        - **Recall** — of the spaghetti plates in the photos, how many it found.
        - **mAP50** — one overall score, from 0 to 1, that counts a box as right when it
          roughly covers the plate. It first passed 0.9 in **round {reached}**, then wobbled before settling.
        - **mAP50-95** — the same score with a much stricter idea of "covers the plate". It is
          always lower; {best['mAP50-95']:.2f} means: finds the plate reliably, draws the box a
          bit loosely.
        """
        if reached
        else ""
    )
    chart = metrics.melt(id_vars="epoch", value_vars=["precision", "recall", "mAP50", "mAP50-95"])
    fig = px.line(chart, x="epoch", y="value", color="variable",
                  labels={"epoch": "round (epoch)", "value": "score (0–1)", "variable": ""})
    fig.add_vline(x=training["best_epoch"], line_dash="dot", line_color="grey",
                  annotation_text=f"kept: round {training['best_epoch']}")
    fig.update_layout(height=380, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "**Reading the chart.** In the first rounds the model draws boxes almost everywhere: it "
        "finds every plate (recall 100 %) but nearly all its boxes are wrong (precision near 0). "
        "Then it learns to be picky. The dips along the way show that training is bumpy — "
        "which is why YOLO keeps the best round rather than the last."
    )
    st.warning(
        f"**Read the near-perfect score with care.** It was measured on only "
        f"{training['val_images']} photos from the same collection as the training photos. "
        "A real deployment would test on photos from the place it will be used — "
        "and the next tab shows where this model still slips."
    )
    with st.expander("📉 The error the model is reducing (loss)"):
        loss = metrics.melt(id_vars="epoch", value_vars=["train box loss", "validation box loss"])
        fig = px.line(loss, x="epoch", y="value", color="variable",
                      labels={"epoch": "round (epoch)", "value": "box error", "variable": ""})
        fig.update_layout(height=320, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "*Loss* is how far the boxes are off. The training error falls steadily. The error on "
            "the held-back photos drops in the first rounds and then jumps around: with only "
            f"{training['val_images']} check photos, one odd photo moves it a lot. More check photos "
            "would help more than more rounds."
        )
    with st.expander("🔎 The maths, if you're curious — when does a box count as right?"):
        st.markdown(
            r"""
            The model's box and the hand-drawn box are compared by their **overlap**, called
            *intersection over union* (IoU):

            $$\text{IoU} = \frac{\text{area both boxes cover}}{\text{area either box covers}}$$

            IoU is 1 for a perfect match and 0 for no overlap. **mAP50** counts a box as right
            from 50 % overlap; **mAP50-95** repeats the scoring at 50 %, 55 %, … 95 % and takes
            the average — so a loosely drawn box loses points there.
            """
        )
        show_source(D.iou, "🔍 Show the overlap calculation")

# ===========================================================================
# 4. Before vs after, including the look-alikes
# ===========================================================================
with tab_compare:
    st.subheader("Same photos, before and after fine-tuning")
    threshold = st.slider(
        "Minimum confidence", 0.05, 0.95, results["draw_confidence"], 0.05, key="compare_threshold"
    )
    st.dataframe(D.compare_models(results, "spaghetti", threshold), hide_index=True, width="stretch")
    key = st.selectbox(
        "Look at one photo", D.images_for(results, "fine_tuned", "spaghetti"),
        format_func=photo_name, key="compare_photo",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Before** — {D.MODEL_NAMES['pretrained']}")
        st.image(D.draw_boxes(D.image_path(key), D.above(pretrained[pretrained["image"] == key], threshold)), width="stretch")
    with c2:
        st.markdown(f"**After** — {D.MODEL_NAMES['fine_tuned']}")
        st.image(D.draw_boxes(D.image_path(key), D.above(fine_tuned[fine_tuned["image"] == key], threshold)), width="stretch")
    def best(model_frame, stem):
        hits = model_frame[model_frame["image"].str.contains(stem)]
        return f"{hits['confidence'].max():.0%}" if len(hits) else "—"

    st.markdown(
        f"""
        **Three photos worth a closer look:**

        - **gnocchi bolognese** — really shell pasta with chicken and cheese. The new model still
          calls it spaghetti, and is *{best(fine_tuned, "gnocchi")} sure*. It learned "pasta in red sauce on a plate", not
          "long thin noodles". High confidence does not mean correct.
        - **pizza and spaghetti** — a pizza topped with spaghetti bolognese. *Before* says pizza,
          *after* says spaghetti — both are right, each with the only words it knows. A model
          can never answer outside its list of labels.
        - **spaghetti hozinja** — besides the plate, the new model draws a weak second box on the
          herbs on the table. Raise the slider to 50 % and it disappears: that is the threshold
          doing its job.

        The fix is the same as with a new colleague: **show it more examples, including the
        look-alikes** — label other pasta dishes as their own class — and test on photos from the
        real setting.
        """
    )

# ===========================================================================
# 5. The code — the point is how little there is
# ===========================================================================
with tab_code:
    cells = D.notebook_cells()
    st.subheader(f"About {D.code_line_count(cells)} lines of code")
    st.markdown(
        """
        This is the complete notebook from the sibling `object-detection/` repo: use the
        pretrained model, fine-tune it, use the fine-tuned model. The hard parts — the network,
        the training loop, the drawing — come with the free **Ultralytics** library. Your work
        is the photos and the labels.
        """
    )
    for kind, source in cells:
        if kind == "code":
            st.code(source, language="python")
        else:  # the notebook's own headings, scaled down to sit inside this tab
            st.markdown(re.sub(r"^#+ ", "#### ", source, flags=re.MULTILINE))
    with st.expander("📄 How the results on this page were exported (export_results.py)"):
        try:
            st.code((SIBLING / "export_results.py").read_text(), language="python")
        except OSError:
            st.warning("Could not read `object-detection/export_results.py`.")

# ===========================================================================
# 6. Licences — worth knowing before you use this at work
# ===========================================================================
with tab_licence:
    st.subheader("Free to learn with — check before you sell with it")
    st.markdown(
        """
        **Ultralytics YOLO** is open source under the **AGPL-3.0** licence, and models you train
        with it fall under it too. For learning, teaching and open-source projects that is fine.
        If you build it into a **closed or commercial product**, you need an
        [Ultralytics Enterprise License](https://www.ultralytics.com/license) — or a detector with
        a more permissive licence (e.g. Apache 2.0). Asking *"what licence is this model under?"*
        is part of choosing a tool at work.

        This app only **shows stored results** — it never runs YOLO. The sibling
        `object-detection/` repo, which does, is itself AGPL-3.0.
        """
    )
    st.markdown("##### Photo credits")
    st.caption("All photos are from Wikimedia Commons, downscaled; boxes drawn by the models above.")
    st.markdown(D.credits_markdown())

st.divider()
with st.container(border=True):
    st.markdown(
        """
🎯 **What to remember**
- **A detector answers three things per object:** where (box), what (label), how sure (confidence).
  The confidence threshold is your business decision.
- **You rarely start from zero.** A free pretrained model plus ~50 labelled photos and a laptop
  were enough to teach it something new.
- **Test it where it will be used.** A near-perfect score on a few similar photos still missed
  the look-alikes — more varied examples are the fix.
"""
    )

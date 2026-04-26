import json
import time
import uuid
from io import BytesIO, StringIO
from typing import Any, Dict, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError
from pypdf import PdfReader


PEGASUS_MODEL_ID_REGIONS = {
    "us-east-1": "us.twelvelabs.pegasus-1-2-v1:0",
    "us-west-2": "us.twelvelabs.pegasus-1-2-v1:0",
    "eu-west-1": "eu.twelvelabs.pegasus-1-2-v1:0",
    "ap-northeast-2": "apac.twelvelabs.pegasus-1-2-v1:0",
}

MARENGO_MODEL_ID = "twelvelabs.marengo-embed-3-0-v1:0"
MARENGO_INFERENCE_ID_REGIONS = {
    "us-east-1": "us.twelvelabs.marengo-embed-3-0-v1:0",
    "us-west-2": "us.twelvelabs.marengo-embed-3-0-v1:0",
    "eu-west-1": "eu.twelvelabs.marengo-embed-3-0-v1:0",
    "ap-northeast-2": "apac.twelvelabs.marengo-embed-3-0-v1:0",
}

SEVERITY_ORDER = {
    "none": 0,
    "minor": 1,
    "moderate": 2,
    "severe": 3,
    "destroyed": 4,
}


@st.cache_resource
def get_aws_clients(
    region_name: Optional[str] = None,
    profile_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
):
    session_kwargs = {"region_name": region_name}

    # If explicit keys are provided, use them. Otherwise, use profile/default chain.
    if aws_access_key_id and aws_secret_access_key:
        session_kwargs["aws_access_key_id"] = aws_access_key_id
        session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            session_kwargs["aws_session_token"] = aws_session_token
    elif profile_name:
        session_kwargs["profile_name"] = profile_name

    session = boto3.Session(**session_kwargs)
    account_id = None
    try:
        account_id = session.client("sts").get_caller_identity()["Account"]
    except (NoCredentialsError, ClientError):
        # Let the app load and show a clear UI message instead of crashing on startup.
        account_id = None

    return {
        "session": session,
        "s3": session.client("s3"),
        "bedrock": session.client("bedrock-runtime"),
        "account_id": account_id,
        "region": session.region_name or region_name,
    }


def upload_to_s3(s3_client, bucket: str, key: str, uploaded_file) -> str:
    uploaded_file.seek(0)
    s3_client.upload_fileobj(uploaded_file, bucket, key)
    return f"s3://{bucket}/{key}"


def read_baseline_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame(columns=["zone", "baseline_condition"])

    if uploaded_file.name.lower().endswith(".csv"):
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(content))
        return normalize_columns(df, {"zone", "baseline_condition"})

    # If baseline is an image, return empty structure. In practice this can be OCR-expanded.
    return pd.DataFrame(columns=["zone", "baseline_condition"])


def read_report_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame(columns=["zone", "reported_damage", "reported_severity"])

    lower_name = uploaded_file.name.lower()

    if lower_name.endswith(".csv"):
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(content))
        return normalize_columns(df, {"zone", "reported_damage", "reported_severity"})

    if lower_name.endswith(".pdf"):
        text = ""
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"

        rows = []
        for line in text.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3 and parts[0].lower() != "zone":
                rows.append(
                    {
                        "zone": parts[0],
                        "reported_damage": parts[1],
                        "reported_severity": parts[2],
                    }
                )

        return pd.DataFrame(rows, columns=["zone", "reported_damage", "reported_severity"])

    return pd.DataFrame(columns=["zone", "reported_damage", "reported_severity"])


def normalize_columns(df: pd.DataFrame, required_cols: set) -> pd.DataFrame:
    renamed = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=renamed)
    missing = required_cols - set(df.columns)
    if missing:
        for col in missing:
            df[col] = ""
    return df[list(required_cols)]


def normalize_severity(value: Any) -> str:
    if pd.isna(value):
        return "none"
    v = str(value).strip().lower()
    if v in SEVERITY_ORDER:
        return v
    if v in {"low"}:
        return "minor"
    if v in {"high"}:
        return "severe"
    return "moderate"


def call_pegasus_findings(
    bedrock_client,
    region: str,
    video_s3_uri: str,
    account_id: Optional[str],
    prompt: str,
) -> Tuple[str, pd.DataFrame]:
    model_id = PEGASUS_MODEL_ID_REGIONS.get(region)
    if not model_id:
        raise ValueError(f"Pegasus is not available in region {region}")

    s3_location = {"uri": video_s3_uri}
    if account_id:
        s3_location["bucketOwner"] = account_id

    request_body = {
        "inputPrompt": prompt,
        "mediaSource": {"s3Location": s3_location},
        "temperature": 0,
        "responseFormat": {
            "jsonSchema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "zone": {"type": "string"},
                        "timestamp": {"type": "string"},
                        "detected_damage": {"type": "string"},
                        "severity": {"type": "string"},
                        "evidence_summary": {"type": "string"},
                    },
                    "required": ["zone", "timestamp", "detected_damage", "severity"],
                },
            }
        },
    }

    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=json.dumps(request_body),
        contentType="application/json",
        accept="application/json",
    )
    parsed = json.loads(response.get("body").read())

    message = parsed.get("message", "[]")
    try:
        findings = json.loads(message)
    except json.JSONDecodeError:
        findings = []

    findings_df = pd.DataFrame(findings)
    if findings_df.empty:
        findings_df = pd.DataFrame(columns=["zone", "timestamp", "detected_damage", "severity", "evidence_summary"])

    if "severity" in findings_df.columns:
        findings_df["severity"] = findings_df["severity"].apply(normalize_severity)

    summary_prompt = "Give a short summary of visible disaster impact in this video in 5 lines max."
    summary_body = {
        "inputPrompt": summary_prompt,
        "mediaSource": {"s3Location": s3_location},
        "temperature": 0,
    }
    summary_response = bedrock_client.invoke_model(
        modelId=model_id,
        body=json.dumps(summary_body),
        contentType="application/json",
        accept="application/json",
    )
    summary_msg = json.loads(summary_response.get("body").read()).get("message", "")

    return summary_msg, findings_df


def wait_for_embedding_output(s3_client, bedrock_client, bucket: str, prefix: str, invocation_arn: str):
    status = None
    for _ in range(72):
        response = bedrock_client.get_async_invoke(invocationArn=invocation_arn)
        status = response.get("status")
        if status in {"Completed", "Failed", "Expired"}:
            break
        time.sleep(5)

    if status != "Completed":
        raise RuntimeError(f"Embedding task failed with status: {status}")

    out = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for obj in out.get("Contents", []):
        if obj["Key"].endswith("output.json"):
            body = s3_client.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read().decode("utf-8")
            return json.loads(body).get("data", [])

    return []


def create_video_embedding(
    s3_client,
    bedrock_client,
    bucket: str,
    embeddings_prefix: str,
    video_s3_uri: str,
    account_id: Optional[str],
):
    unique_id = str(uuid.uuid4())
    out_prefix = f"{embeddings_prefix.rstrip('/')}/videos/{unique_id}"

    s3_location = {"uri": video_s3_uri}
    if account_id:
        s3_location["bucketOwner"] = account_id

    response = bedrock_client.start_async_invoke(
        modelId=MARENGO_MODEL_ID,
        modelInput={
            "inputType": "video",
            "video": {
                "mediaSource": {"s3Location": s3_location},
                "embeddingOption": ["visual"],
                "embeddingScope": ["clip"],
            },
        },
        outputDataConfig={"s3OutputDataConfig": {"s3Uri": f"s3://{bucket}/{out_prefix}/"}},
    )
    arn = response["invocationArn"]
    return wait_for_embedding_output(s3_client, bedrock_client, bucket, out_prefix, arn)


def create_text_embedding(bedrock_client, region: str, text_query: str):
    model_id = MARENGO_INFERENCE_ID_REGIONS.get(region)
    if not model_id:
        raise ValueError(f"Marengo inference profile not available in region {region}")

    model_input = {"inputType": "text", "text": {"inputText": text_query}}
    response = bedrock_client.invoke_model(modelId=model_id, body=json.dumps(model_input))
    data = json.loads(response["body"].read().decode("utf-8")).get("data", [])
    return data


def search_damage_moments(bedrock_client, region: str, video_embeddings: List[Dict[str, Any]], query: str, top_k: int = 5):
    text_embedding_data = create_text_embedding(bedrock_client, region, query)
    if not text_embedding_data:
        return pd.DataFrame(columns=["start_time", "end_time", "score"])

    query_vec = np.array(text_embedding_data[0]["embedding"])
    rows = []
    for segment in video_embeddings:
        seg_vec = np.array(segment.get("embedding", []))
        if seg_vec.size == 0:
            continue
        score = float(np.dot(query_vec, seg_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(seg_vec)))
        rows.append(
            {
                "start_time": segment.get("startSec"),
                "end_time": segment.get("endSec"),
                "score": round(score, 4),
                "embedding_option": segment.get("embeddingOption", "visual"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["start_time", "end_time", "score"])

    return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k)


def classify_row(row: pd.Series) -> str:
    video = normalize_severity(row.get("severity", "none"))
    report = normalize_severity(row.get("reported_severity", "none"))

    if video == report:
        return "validated report"
    if report == "none" and video != "none":
        return "unreported damage"
    if SEVERITY_ORDER[video] > SEVERITY_ORDER[report]:
        return "underestimated damage"
    return "mismatch"


def build_fusion(video_df: pd.DataFrame, baseline_df: pd.DataFrame, report_df: pd.DataFrame) -> pd.DataFrame:
    if video_df.empty:
        return pd.DataFrame(columns=["zone", "fusion_status"])

    for col in ["zone", "timestamp", "detected_damage", "severity", "evidence_summary"]:
        if col not in video_df.columns:
            video_df[col] = ""

    merged = (
        video_df.merge(baseline_df, on="zone", how="left")
        .merge(report_df, on="zone", how="left")
        .copy()
    )

    merged["severity"] = merged["severity"].apply(normalize_severity)
    if "reported_severity" in merged.columns:
        merged["reported_severity"] = merged["reported_severity"].apply(normalize_severity)
    else:
        merged["reported_severity"] = "none"

    merged["fusion_status"] = merged.apply(classify_row, axis=1)
    merged["final_severity"] = merged[["severity", "reported_severity"]].apply(
        lambda r: max(r, key=lambda x: SEVERITY_ORDER.get(x, 0)), axis=1
    )

    merged["evidence_chain"] = merged.apply(
        lambda r: f"zone={r.get('zone','')}; ts={r.get('timestamp','')}; detected={r.get('detected_damage','')}",
        axis=1,
    )

    return merged


def dataframe_to_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def main():
    st.set_page_config(page_title="Track 3 Disaster Assessment", layout="wide")
    st.title("Track 3 Disaster Assessment Streamlit App")
    st.caption("Upload -> S3 -> Pegasus/Marengo -> Fusion -> Outputs")

    with st.sidebar:
        st.header("AWS Configuration")
        region_override = st.text_input("AWS Region (optional override)", value="us-east-1")
        profile_name = st.text_input("AWS Profile (optional)", value="")

        with st.expander("Or enter temporary AWS keys", expanded=False):
            aws_access_key_id = st.text_input("AWS Access Key ID", value="", type="password")
            aws_secret_access_key = st.text_input("AWS Secret Access Key", value="", type="password")
            aws_session_token = st.text_input("AWS Session Token (optional)", value="", type="password")

        bucket = st.text_input("S3 Bucket", value="")
        videos_prefix = st.text_input("Videos Prefix", value="videos")
        baseline_prefix = st.text_input("Baseline Prefix", value="baseline")
        reports_prefix = st.text_input("Reports Prefix", value="reports")
        embeddings_prefix = st.text_input("Embeddings Prefix", value="embeddings")

    clients = get_aws_clients(
        region_override or None,
        profile_name or None,
        aws_access_key_id or None,
        aws_secret_access_key or None,
        aws_session_token or None,
    )
    s3_client = clients["s3"]
    bedrock_client = clients["bedrock"]
    account_id = clients["account_id"]
    region = clients["region"]

    if account_id:
        st.info(f"Active AWS region: {region} | Account: {account_id}")
    else:
        st.warning(
            "AWS credentials are not configured. The app loaded, but upload/analysis actions will fail until credentials are available. "
            "Configure AWS credentials (aws configure / AWS profile / environment variables), then refresh the app."
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        video_file = st.file_uploader("Upload post-disaster video", type=["mp4", "mov", "mkv", "avi", "webm"])
    with c2:
        baseline_file = st.file_uploader("Upload baseline CSV/image", type=["csv", "png", "jpg", "jpeg"])
    with c3:
        report_file = st.file_uploader("Upload damage report CSV/PDF", type=["csv", "pdf"])

    if st.button("Upload Files to S3", type="primary"):
        if not bucket:
            st.error("Please provide S3 bucket in sidebar.")
        elif not account_id:
            st.error("AWS credentials not found. Configure credentials first, then retry upload.")
        else:
            try:
                uploaded_keys = {}

                if video_file is not None:
                    v_key = f"{videos_prefix.rstrip('/')}/{video_file.name}"
                    v_uri = upload_to_s3(s3_client, bucket, v_key, video_file)
                    uploaded_keys["video_s3_uri"] = v_uri
                    st.success(f"Uploaded video: {v_uri}")

                if baseline_file is not None:
                    b_key = f"{baseline_prefix.rstrip('/')}/{baseline_file.name}"
                    b_uri = upload_to_s3(s3_client, bucket, b_key, baseline_file)
                    uploaded_keys["baseline_s3_uri"] = b_uri
                    st.success(f"Uploaded baseline: {b_uri}")

                if report_file is not None:
                    r_key = f"{reports_prefix.rstrip('/')}/{report_file.name}"
                    r_uri = upload_to_s3(s3_client, bucket, r_key, report_file)
                    uploaded_keys["report_s3_uri"] = r_uri
                    st.success(f"Uploaded report: {r_uri}")

                st.session_state.update(uploaded_keys)
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    st.subheader("Video Intelligence")
    prompt = st.text_area(
        "Pegasus prompt",
        value=(
            "Analyze this disaster video and return JSON array with: "
            "zone, timestamp, detected_damage, severity, evidence_summary."
        ),
        height=100,
    )

    damage_query = st.text_input("Marengo search query", value="collapsed buildings and severe flooding")

    if st.button("Run Pegasus + Marengo"):
        video_s3_uri = st.session_state.get("video_s3_uri")
        if not video_s3_uri:
            st.error("Upload a video first.")
        elif not bucket:
            st.error("Set S3 bucket first.")
        elif not account_id:
            st.error("AWS credentials not found. Configure credentials first, then run analysis.")
        elif not region:
            st.error("AWS region could not be determined. Set region in the sidebar and retry.")
        else:
            try:
                with st.spinner("Running Pegasus..."):
                    summary, video_findings_df = call_pegasus_findings(
                        bedrock_client, region, video_s3_uri, account_id, prompt
                    )

                st.session_state["video_findings_df"] = video_findings_df
                st.session_state["video_summary"] = summary

                st.markdown("### Pegasus Summary")
                st.write(summary)

                st.markdown("### Pegasus Findings")
                st.dataframe(video_findings_df, use_container_width=True)

                with st.spinner("Creating Marengo video embeddings..."):
                    video_embedding_data = create_video_embedding(
                        s3_client,
                        bedrock_client,
                        bucket,
                        embeddings_prefix,
                        video_s3_uri,
                        account_id,
                    )

                st.session_state["video_embedding_data"] = video_embedding_data

                with st.spinner("Searching damage moments with Marengo..."):
                    moments_df = search_damage_moments(
                        bedrock_client,
                        region,
                        video_embedding_data,
                        damage_query,
                        top_k=5,
                    )

                st.session_state["moments_df"] = moments_df
                st.markdown("### Marengo Damage Moments")
                st.dataframe(moments_df, use_container_width=True)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    st.subheader("Fusion Layer")
    baseline_df = read_baseline_file(baseline_file)
    report_df = read_report_file(report_file)

    if st.button("Run Fusion and Generate Outputs"):
        video_findings_df = st.session_state.get("video_findings_df", pd.DataFrame())
        fusion_df = build_fusion(video_findings_df, baseline_df, report_df)
        st.session_state["fusion_df"] = fusion_df

        severity_table = fusion_df[["zone", "final_severity", "fusion_status"]].copy() if not fusion_df.empty else fusion_df
        evidence_chain = fusion_df[["zone", "timestamp", "detected_damage", "evidence_chain"]].copy() if not fusion_df.empty else fusion_df

        payload = {
            "video_summary": st.session_state.get("video_summary", ""),
            "severity_table": severity_table.to_dict(orient="records") if not severity_table.empty else [],
            "evidence_chain": evidence_chain.to_dict(orient="records") if not evidence_chain.empty else [],
            "fusion_results": fusion_df.to_dict(orient="records") if not fusion_df.empty else [],
        }

        st.markdown("### Severity Table")
        st.dataframe(severity_table, use_container_width=True)

        st.markdown("### Evidence Chain")
        st.dataframe(evidence_chain, use_container_width=True)

        st.markdown("### Final JSON")
        st.json(payload)

        report_text = [
            "Disaster Assessment Report",
            "",
            f"Summary: {st.session_state.get('video_summary', '')}",
            "",
            "Zone outcomes:",
        ]
        for _, row in severity_table.iterrows() if not severity_table.empty else []:
            report_text.append(
                f"- {row['zone']}: severity={row['final_severity']}, status={row['fusion_status']}"
            )

        report_content = "\n".join(report_text)
        st.markdown("### Narrative Report")
        st.text(report_content)

        st.download_button(
            "Download Severity Table (CSV)",
            data=dataframe_to_download(severity_table),
            file_name="severity_table.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download Evidence Chain (CSV)",
            data=dataframe_to_download(evidence_chain),
            file_name="evidence_chain.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download Final JSON",
            data=json.dumps(payload, indent=2).encode("utf-8"),
            file_name="final_assessment.json",
            mime="application/json",
        )
        st.download_button(
            "Download Disaster Assessment Report (TXT)",
            data=report_content.encode("utf-8"),
            file_name="disaster_assessment_report.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()

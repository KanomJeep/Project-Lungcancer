import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE


st.set_page_config(page_title="Data visualization การจำแนกการเสียชีวิตจากโรคมะเร็งปอด", layout="wide", page_icon="📢")

# Inject CSS
st.markdown("""
    <style>
    /* เปลี่ยนสีพื้นหลังหลัก */
    .stApp {
        background-color: #f0f8ff; /* AliceBlue สีฟ้าอ่อนๆ */
    }
    
    /* ปรับหัวข้อ (Header) เป็นสีน้ำเงินเข้ม */
    h1, h2, h3 {
        color: #005b96 !important; /* Navy Blue */
        font-family: 'Sarabun', sans-serif;
    }
    
    /* ปรับแต่ง Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 2px solid #e6e6e6;
    }
    
    /* ปรับแต่งปุ่มกด */
    .stButton>button {
        background-color: #007bff; /* Bootstrap Blue */
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    
    /* กรอบข้อมูล */
    .css-1r6slb0 {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ส่วนหัวของโปรแกรม (Header)
col_head1, col_head2 = st.columns([1, 15])
with col_head1:
    st.markdown("# 📢")
with col_head2:
    st.title("Data visualization การจำแนกการเสียชีวิตจากโรคมะเร็งปอด")
    st.markdown("**Lung Cancer Data visualization System** | *(Demo System)*")

st.markdown("---")

# Sidebar: ส่วนอัปโหลดข้อมูล
st.sidebar.header("📂 นำเข้าข้อมูลคนไข้")
uploaded_file = st.sidebar.file_uploader("Upload CSV File (Lung Cancer.csv)", type=["csv"])

if uploaded_file is not None:
    # โหลดข้อมูล
    df = pd.read_csv(uploaded_file)
    df_raw = df.copy() # เก็บข้อมูลดิบไว้ใช้วิเคราะห์ตอนท้าย

    # ส่วนสำรวจข้อมูลดิบ (EDA)
    st.header("1. การสำรวจข้อมูลพื้นฐาน (Data Exploration)")
    
    tab1, tab2 = st.tabs(["📋 ตารางข้อมูล", "📊 กราฟสรุปผล"])

    with tab1:
        st.subheader("ตัวอย่างข้อมูลคนไข้")
        st.dataframe(df_raw.head())
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.write(f"**จำนวนคนไข้ทั้งหมด:** {df_raw.shape[0]} คน")
        with col_stat2:
            st.write("**ตรวจสอบค่าว่าง:**")
            st.dataframe(df_raw.isnull().sum().to_frame(name='Missing').T)

    with tab2:
        st.subheader("การกระจายตัวของข้อมูล")
        cat_cols_viz = [c for c in df_raw.columns if df_raw[c].dtype == 'object' or len(df_raw[c].unique()) < 10]
        if 'id' in cat_cols_viz: cat_cols_viz.remove('id')
        
        selected_col_viz = st.selectbox("เลือกตัวแปรที่ต้องการดูกราฟ:", cat_cols_viz, index=0)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df_raw, x=selected_col_viz, palette="Blues_d", ax=ax)
        
        plt.title(f"Distribution of {selected_col_viz}") 
        plt.xlabel(selected_col_viz) 
        plt.ylabel("Count") 
        plt.xticks(rotation=0) 
        st.pyplot(fig)

    st.markdown("---")

    # การเตรียมข้อมูล (Preprocessing)
    st.header("2. การเตรียมข้อมูล (Preprocessing)")
    st.info("ℹ️ ระบบกำลังทำการแปลงข้อมูลข้อความ (Text) ให้เป็นตัวเลข (Numeric) เพื่อใช้ในการคำนวณ")
    
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # กำหนดคอลัมน์
    categorical_cols = ['gender', 'country', 'diagnosis_date', 'cancer_stage', 
                        'family_history', 'smoking_status', 'treatment_type', 
                        'end_treatment_date']
    
    le = LabelEncoder()
    # Loop แปลงข้อมูล
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    
    if 'cancer_stage' in df.columns:
        df['cancer_stage'] = df['cancer_stage'] + 1
        
    # --- แสดงตารางเปรียบเทียบ Before / After ---
    col_trans1, col_trans2 = st.columns(2)
    
    with col_trans1:
        st.markdown("#### 📄 ก่อนแปลง (Before Transform)")
        # เลือกเฉพาะคอลัมน์ที่มีการแปลงเพื่อแสดงผลให้ชัดเจน
        cols_to_show = [c for c in categorical_cols if c in df_raw.columns]
        if cols_to_show:
             st.dataframe(df_raw[cols_to_show].head(5))
        else:
             st.dataframe(df_raw.head(5))

    with col_trans2:
        st.markdown("#### 🔢 หลังแปลง (After Transform)")
        cols_to_show = [c for c in categorical_cols if c in df.columns]
        if cols_to_show:
             st.dataframe(df[cols_to_show].head(5))
        else:
             st.dataframe(df.head(5))
             
    st.success("✅ แปลงข้อมูล Text เป็นตัวเลขเรียบร้อย")

    target_col = 'survived'
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # 5. SMOTE
        st.header("3. การแก้ปัญหาข้อมูลไม่สมดุล (SMOTE)")
        col_smote1, col_smote2 = st.columns(2)
        
        with col_smote1:
            st.write("#### 🔴 ก่อนทำ SMOTE")
            fig, ax = plt.subplots(figsize=(5,3))
            y.value_counts().plot(kind='bar', ax=ax, color=['#b0c4de','#4682b4'])
            
            plt.title("Class Distribution (Before SMOTE)")
            plt.xlabel("Class (0=Deceased, 1=Survived)")
            plt.ylabel("Count")
            plt.xticks(rotation=0) 
            st.pyplot(fig)
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        with col_smote2:
            st.write("#### 🔵 หลังทำ SMOTE")
            fig, ax = plt.subplots(figsize=(5,3))
            y_resampled.value_counts().plot(kind='bar', ax=ax, color=['#4682b4','#4682b4'])
            
            plt.title("Class Distribution (After SMOTE)")
            plt.xlabel("Class (0=Deceased, 1=Survived)")
            plt.ylabel("Count")
            plt.xticks(rotation=0)
            st.pyplot(fig)

        # Feature Selection
        st.header("4. การเลือกปัจจัยเสี่ยงสำคัญ (Feature Importance)")
        
        if st.checkbox("แสดงกราฟความสำคัญของฟีเจอร์"):
            mutual_info = mutual_info_classif(X_resampled, y_resampled, random_state=42)
            mutual_info = pd.Series(mutual_info, index=X.columns).sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            mutual_info.plot.bar(ax=ax, color='#007bff')
            
            plt.title("Feature Importance Score")
            plt.ylabel("Mutual Information Score")
            plt.xlabel("Features")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

        # Model Evaluation
        st.header("5. การประมวลผลโมเดล (Model Prediction)")
        
        col_param1, col_param2 = st.columns(2)
        with col_param1:
            test_size = st.slider("Test Size %", 10, 50, 20)
        with col_param2:
            model_name = st.selectbox("Select Model", ["KNN", "Decision Tree", "Naive Bayes"])
        
        if st.button("เริ่มการทำนายผล"):
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=test_size/100, random_state=42
            )
            
            if "KNN" in model_name:
                model = KNeighborsClassifier(n_neighbors=3)
            elif "Decision Tree" in model_name:
                model = DecisionTreeClassifier(random_state=42)
            else:
                model = GaussianNB()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            st.success(f"ความแม่นยำ (Accuracy): {acc:.2%}")
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.write("**Confusion Matrix:**")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4,3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                plt.title('Confusion Matrix')
                st.pyplot(fig)
            with col_res2:
                st.write("**Report:**")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

        st.markdown("---")

        # New Section: Detailed Analysis
        st.header("6. วิเคราะห์เจาะลึกปัจจัยการรอดชีวิต (In-depth Survival Analysis)")
        st.markdown("ส่วนนี้วิเคราะห์ความสัมพันธ์ระหว่างตัวแปรต่างๆ กับการรอดชีวิต (จากข้อมูลจริง)")

        # เตรียมข้อมูลสำหรับการ Plot
        df_analysis = df_raw.copy()
        if 'survived' in df_analysis.columns:
            # ใช้ Label ภาษาอังกฤษสำหรับกราฟ
            df_analysis['Survival_Label'] = df_analysis['survived'].map({0: 'Deceased', 1: 'Survived'})
            
            # เลือกหัวข้อที่จะวิเคราะห์
            analysis_topic = st.selectbox(
                "เลือกปัจจัยที่ต้องการวิเคราะห์ (Select Factor):",
                ['gender', 'smoking_status', 'cancer_stage', 'treatment_type', 'country']
            )
            
            col_an1, col_an2 = st.columns([2, 1])
            
            with col_an1:
                # กราฟแท่งเปรียบเทียบ
                st.subheader(f"กราฟแสดงผล: {analysis_topic} vs Survival")
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # ใช้ countplot แบบ hue
                sns.countplot(data=df_analysis, x=analysis_topic, hue='Survival_Label', palette=['#ff9999', '#66b3ff'], ax=ax)
                
                # --- กราฟภาษาอังกฤษ ---
                plt.title(f"Survival Status by {analysis_topic}")
                plt.xlabel(analysis_topic)
                plt.ylabel("Count")
                plt.legend(title='Status')
                plt.xticks(rotation=0)
                st.pyplot(fig)
            
            with col_an2:
                # คำนวณ % การรอดชีวิต
                st.subheader("💡 สรุปผลวิเคราะห์")
                
                # Group by เพื่อหา %
                summary = df_analysis.groupby(analysis_topic)['survived'].mean() * 100
                summary_count = df_analysis.groupby(analysis_topic)['survived'].count()
                
                st.write(f"**อัตราการรอดชีวิตตามกลุ่ม (Survival Rate %):**")
                for category in summary.index:
                    rate = summary[category]
                    count = summary_count[category]
                    st.write(f"- **{category}**: {rate:.2f}% (จาก {count} คน)")
                
                # ไฮไลท์ข้อมูล
                best_group = summary.idxmax()
                worst_group = summary.idxmin()
                
                st.info(f"🏆 กลุ่มที่มีโอกาสรอดสูงสุด: **{best_group}** ({summary.max():.2f}%)")
                st.warning(f"⚠️ กลุ่มที่มีความเสี่ยงสูงสุด: **{worst_group}** ({summary.min():.2f}%)")
            
            # เพิ่มเติม: กราฟวงกลมสำหรับดูสัดส่วนรวม
            st.write("")
            with st.expander(f"ดูรายละเอียดสัดส่วนแบบกราฟวงกลม (Pie Chart) ของ {analysis_topic}"):
                 # สร้าง Pie Chart แยกตามกลุ่ม
                unique_vals = df_analysis[analysis_topic].unique()
                cols = st.columns(len(unique_vals))
                
                for i, val in enumerate(unique_vals):
                    with cols[i]:
                        subset = df_analysis[df_analysis[analysis_topic] == val]
                        
                        surv_counts = subset['Survival_Label'].value_counts()
                        
                        if not surv_counts.empty:
                            fig, ax = plt.subplots(figsize=(3,3))
                            ax.pie(surv_counts, labels=surv_counts.index, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'], startangle=90)
                            
                            ax.set_title(f"Group: {val}")
                            st.pyplot(fig)

    else:
        st.error("ไม่พบข้อมูล 'survived' สำหรับวิเคราะห์")

else:
    st.info("👈 กรุณาอัปโหลดไฟล์ข้อมูล (CSV) ที่แถบเมนูด้านซ้าย")
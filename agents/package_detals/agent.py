from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Annotated, Literal


# === 1) Tool input schema ===
class Policy(BaseModel):
    # swapped "platinum" -> "diamond" to match your sheets
    package_type: Literal["basic", "bronze", "silver", "gold", "diamond"] = Field(
        default="basic",
        description="The name of the policy package. Must be one of: basic, bronze, silver, gold, diamond."
    )


insurance_data = {
  "insurance_packages": [
    {
      "package_type": "A. Basic Package",
      "general_info": {
        "Product": "A. Basic Package",
        "Network": "M",
        "Annual Limit": "SAR 500,000",
        "Room": "Shared",
        "Deductible%": "20%",
        "Hospital deductible limit (Co-Pay)": "SAR 300",
        "Optical Limit": "SAR 400",
        "Dental deductible": "20%",
        "Dental Limit": "SAR 2,000",
        "Metarnity": "SAR 15,000"
      },
      "benefits": [
        {
          "Benefit": "Hospital Admission",
          "Benefit Description": "Inpatient care, surgeries, one-day operations, maternity",
          "Covered Services": "Shared room, surgeries, pregnancy, childbirth",
          "Limit (SAR)": "Covered, shared room"
        },
        {
          "Benefit": "Emergency Treatment",
          "Benefit Description": "Urgent care for life-threatening or disabling conditions",
          "Covered Services": "Resuscitation, emergency surgery, trauma",
          "Limit (SAR)": "Full coverage"
        },
        {
          "Benefit": "Outpatient Services",
          "Benefit Description": "General consultations, referrals, lab, x-rays",
          "Covered Services": "All specialties, family medicine, pediatrics",
          "Limit (SAR)": "Deductible 20%, Co-pay SAR 300"
        },
        {
          "Benefit": "Medication",
          "Benefit Description": "Coverage for prescription drugs",
          "Covered Services": "Generics: 20% copay (SAR 30 max); Brands (if no generic): 0–50%",
          "Limit (SAR)": "Per IDF policy"
        },
        {
          "Benefit": "Mental Health",
          "Benefit Description": "Treatment of psychological conditions",
          "Covered Services": "Depression, anxiety, schizophrenia, PTSD, autism, Alzheimer's",
          "Limit (SAR)": "Up to SAR 50,000 (where applicable)"
        },
        {
          "Benefit": "Maternity and Childcare",
          "Benefit Description": "Pre/postnatal care, childbirth, infant coverage",
          "Covered Services": "Delivery + complications, newborn up to 30 days",
          "Limit (SAR)": "SAR 15,000"
        },
        {
          "Benefit": "Newborn Screening",
          "Benefit Description": "Early detection of congenital disabilities",
          "Covered Services": "Critical congenital heart disease, metabolic screening",
          "Limit (SAR)": "SAR 100,000"
        },
        {
          "Benefit": "Vaccinations",
          "Benefit Description": "Preventive immunizations",
          "Covered Services": "Per MoH vaccination schedule for adults and children",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Dental Care",
          "Benefit Description": "Basic dental treatments",
          "Covered Services": "Consultations, cleaning, fillings, extractions, periodontal care",
          "Limit (SAR)": "SAR 2,000; 20% deductible"
        },
        {
          "Benefit": "Optical Care",
          "Benefit Description": "Vision correction",
          "Covered Services": "Eyeglasses for <14 yrs",
          "Limit (SAR)": "SAR 400"
        },
        {
          "Benefit": "Bariatric Surgery",
          "Benefit Description": "Obesity-related surgeries",
          "Covered Services": "If BMI ≥ 40 or ≥ 35 with comorbidities",
          "Limit (SAR)": "SAR 15,000 (as per national cap)"
        },
        {
          "Benefit": "Contraception",
          "Benefit Description": "Family planning services",
          "Covered Services": "Hormonal treatment, IUDs",
          "Limit (SAR)": "SAR 1,500"
        },
        {
          "Benefit": "Dialysis",
          "Benefit Description": "Renal replacement therapy",
          "Covered Services": "In-center, peritoneal, or home dialysis",
          "Limit (SAR)": "SAR 180,000"
        },
        {
          "Benefit": "Kidney Transplant",
          "Benefit Description": "ESRD organ transplant",
          "Covered Services": "Recipient and donor procedure costs",
          "Limit (SAR)": "SAR 250,000"
        },
        {
          "Benefit": "Organ Harvest (Donor)",
          "Benefit Description": "Donor surgery",
          "Covered Services": "For insured donor",
          "Limit (SAR)": "SAR 50,000"
        },
        {
          "Benefit": "Home Healthcare",
          "Benefit Description": "At-home follow-up care",
          "Covered Services": "Post-surgery wound care, IV therapy, catheter support",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Telemedicine",
          "Benefit Description": "Remote consultations",
          "Covered Services": "MoH-licensed providers",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Medical Equipment",
          "Benefit Description": "Medical monitoring tools",
          "Covered Services": "Glucose monitors, insulin pumps, BP monitors, hearing aids",
          "Limit (SAR)": "Hearing aids SAR 6,000; others per policy"
        },
        {
          "Benefit": "Disability Treatment",
          "Benefit Description": "Rehabilitation and assistive support",
          "Covered Services": "Physical therapy, speech therapy, occupational therapy",
          "Limit (SAR)": "SAR 100,000"
        },
        {
          "Benefit": "Congenital/Hereditary Conditions",
          "Benefit Description": "Genetic or birth-related disorders",
          "Covered Services": "All congenital defects with life-threatening implications",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Circumcision (Male)",
          "Benefit Description": "Infant male circumcision",
          "Covered Services": "As per medical indication",
          "Limit (SAR)": "SAR 500"
        },
        {
          "Benefit": "Corpse Repatriation",
          "Benefit Description": "Repatriation of deceased to home country",
          "Covered Services": "Preparation and transport",
          "Limit (SAR)": "SAR 10,000"
        },
        {
          "Benefit": "Complications Coverage",
          "Benefit Description": "Post-treatment medical issues",
          "Covered Services": "From approved procedures",
          "Limit (SAR)": "Within overall policy cap"
        }
      ]
    },
    {
      "package_type": "B. Bronze Package",
      "general_info": {
        "Product": "B. Bronze Package",
        "Network": "R",
        "Annual Limit": "SAR 500,000",
        "Room": "Shared",
        "Deductible%": "20%",
        "Hospital deductible limit (Co-Pay)": "SAR 300",
        "Optical Limit": "SAR 400",
        "Dental deductible": "20%",
        "Dental Limit": "SAR 2,000",
        "Metarnity": "SAR 15,000"
      },
      "benefits": [
        {
          "Benefit": "Hospital Admission",
          "Benefit Description": "Inpatient care, surgeries, one-day operations, maternity",
          "Covered Services": "Shared room, surgeries, pregnancy, childbirth",
          "Limit (SAR)": "Covered, shared room"
        },
        {
          "Benefit": "Emergency Treatment",
          "Benefit Description": "Urgent care for life-threatening or disabling conditions",
          "Covered Services": "Resuscitation, emergency surgery, trauma",
          "Limit (SAR)": "Full coverage"
        },
        {
          "Benefit": "Outpatient Services",
          "Benefit Description": "General consultations, referrals, lab, x-rays",
          "Covered Services": "All specialties, family medicine, pediatrics",
          "Limit (SAR)": "Deductible 20%, Co-pay SAR 300"
        },
        {
          "Benefit": "Medication",
          "Benefit Description": "Coverage for prescription drugs",
          "Covered Services": "Generics: 20% copay (SAR 30 max); Brands (if no generic): 0–50%",
          "Limit (SAR)": "Per IDF policy"
        },
        {
          "Benefit": "Mental Health",
          "Benefit Description": "Treatment of psychological conditions",
          "Covered Services": "Depression, anxiety, schizophrenia, PTSD, autism, Alzheimer's",
          "Limit (SAR)": "Up to SAR 50,000 (where applicable)"
        },
        {
          "Benefit": "Maternity and Childcare",
          "Benefit Description": "Pre/postnatal care, childbirth, infant coverage",
          "Covered Services": "Delivery + complications, newborn up to 30 days",
          "Limit (SAR)": "SAR 15,000"
        },
        {
          "Benefit": "Newborn Screening",
          "Benefit Description": "Early detection of congenital disabilities",
          "Covered Services": "Critical congenital heart disease, metabolic screening",
          "Limit (SAR)": "SAR 100,000"
        },
        {
          "Benefit": "Vaccinations",
          "Benefit Description": "Preventive immunizations",
          "Covered Services": "Per MoH vaccination schedule for adults and children",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Dental Care",
          "Benefit Description": "Basic dental treatments",
          "Covered Services": "Consultations, cleaning, fillings, extractions, periodontal care",
          "Limit (SAR)": "SAR 2,000; 20% deductible"
        },
        {
          "Benefit": "Optical Care",
          "Benefit Description": "Vision correction",
          "Covered Services": "Eyeglasses for <14 yrs",
          "Limit (SAR)": "SAR 400"
        },
        {
          "Benefit": "Bariatric Surgery",
          "Benefit Description": "Obesity-related surgeries",
          "Covered Services": "If BMI ≥ 40 or ≥ 35 with comorbidities",
          "Limit (SAR)": "SAR 15,000 (as per national cap)"
        },
        {
          "Benefit": "Contraception",
          "Benefit Description": "Family planning services",
          "Covered Services": "Hormonal treatment, IUDs",
          "Limit (SAR)": "SAR 1,500"
        },
        {
          "Benefit": "Dialysis",
          "Benefit Description": "Renal replacement therapy",
          "Covered Services": "In-center, peritoneal, or home dialysis",
          "Limit (SAR)": "SAR 180,000"
        },
        {
          "Benefit": "Kidney Transplant",
          "Benefit Description": "ESRD organ transplant",
          "Covered Services": "Recipient and donor procedure costs",
          "Limit (SAR)": "SAR 250,000"
        },
        {
          "Benefit": "Organ Harvest (Donor)",
          "Benefit Description": "Donor surgery",
          "Covered Services": "For insured donor",
          "Limit (SAR)": "SAR 50,000"
        },
        {
          "Benefit": "Home Healthcare",
          "Benefit Description": "At-home follow-up care",
          "Covered Services": "Post-surgery wound care, IV therapy, catheter support",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Telemedicine",
          "Benefit Description": "Remote consultations",
          "Covered Services": "MoH-licensed providers",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Medical Equipment",
          "Benefit Description": "Medical monitoring tools",
          "Covered Services": "Glucose monitors, insulin pumps, BP monitors, hearing aids",
          "Limit (SAR)": "Hearing aids SAR 6,000; others per policy"
        },
        {
          "Benefit": "Disability Treatment",
          "Benefit Description": "Rehabilitation and assistive support",
          "Covered Services": "Physical therapy, speech therapy, occupational therapy",
          "Limit (SAR)": "SAR 100,000"
        },
        {
          "Benefit": "Congenital/Hereditary Conditions",
          "Benefit Description": "Genetic or birth-related disorders",
          "Covered Services": "All congenital defects with life-threatening implications",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Circumcision (Male)",
          "Benefit Description": "Infant male circumcision",
          "Covered Services": "As per medical indication",
          "Limit (SAR)": "SAR 500"
        },
        {
          "Benefit": "Corpse Repatriation",
          "Benefit Description": "Repatriation of deceased to home country",
          "Covered Services": "Preparation and transport",
          "Limit (SAR)": "SAR 10,000"
        },
        {
          "Benefit": "Complications Coverage",
          "Benefit Description": "Post-treatment medical issues",
          "Covered Services": "From approved procedures",
          "Limit (SAR)": "Within overall policy cap"
        }
      ]
    },
    {
      "package_type": "B. Bronze Package",
      "general_info": {
        "Product": "B. Bronze Package",
        "Network": "S",
        "Annual Limit": "SAR 500,000",
        "Room": "Shared",
        "Deductible%": "20%",
        "Hospital deductible limit (Co-Pay)": "SAR 300",
        "Optical Limit": "SAR 400",
        "Dental deductible": "20%",
        "Dental Limit": "SAR 2,000",
        "Metarnity": "SAR 15,000"
      },
      "benefits": [
        {
          "Benefit": "Hospital Admission",
          "Benefit Description": "Inpatient care, surgeries, one-day operations, maternity",
          "Covered Services": "Shared room, surgeries, pregnancy, childbirth",
          "Limit (SAR)": "Covered, shared room"
        },
        {
          "Benefit": "Emergency Treatment",
          "Benefit Description": "Urgent care for life-threatening or disabling conditions",
          "Covered Services": "Resuscitation, emergency surgery, trauma",
          "Limit (SAR)": "Full coverage"
        },
        {
          "Benefit": "Outpatient Services",
          "Benefit Description": "General consultations, referrals, lab, x-rays",
          "Covered Services": "All specialties, family medicine, pediatrics",
          "Limit (SAR)": "Deductible 20%, Co-pay SAR 300"
        },
        {
          "Benefit": "Medication",
          "Benefit Description": "Coverage for prescription drugs",
          "Covered Services": "Generics: 20% copay (SAR 30 max); Brands (if no generic): 0–50%",
          "Limit (SAR)": "Per IDF policy"
        },
        {
          "Benefit": "Mental Health",
          "Benefit Description": "Treatment of psychological conditions",
          "Covered Services": "Depression, anxiety, schizophrenia, PTSD, autism, Alzheimer's",
          "Limit (SAR)": "Up to SAR 50,000 (where applicable)"
        },
        {
          "Benefit": "Maternity and Childcare",
          "Benefit Description": "Pre/postnatal care, childbirth, infant coverage",
          "Covered Services": "Delivery + complications, newborn up to 30 days",
          "Limit (SAR)": "SAR 15,000"
        },
        {
          "Benefit": "Newborn Screening",
          "Benefit Description": "Early detection of congenital disabilities",
          "Covered Services": "Critical congenital heart disease, metabolic screening",
          "Limit (SAR)": "SAR 100,000"
        },
        {
          "Benefit": "Vaccinations",
          "Benefit Description": "Preventive immunizations",
          "Covered Services": "Per MoH vaccination schedule for adults and children",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Dental Care",
          "Benefit Description": "Basic dental treatments",
          "Covered Services": "Consultations, cleaning, fillings, extractions, periodontal care",
          "Limit (SAR)": "SAR 2,000; 20% deductible"
        },
        {
          "Benefit": "Optical Care",
          "Benefit Description": "Vision correction",
          "Covered Services": "Eyeglasses for <14 yrs",
          "Limit (SAR)": "SAR 400"
        },
        {
          "Benefit": "Bariatric Surgery",
          "Benefit Description": "Obesity-related surgeries",
          "Covered Services": "If BMI ≥ 40 or ≥ 35 with comorbidities",
          "Limit (SAR)": "SAR 15,000 (as per national cap)"
        },
        {
          "Benefit": "Contraception",
          "Benefit Description": "Family planning services",
          "Covered Services": "Hormonal treatment, IUDs",
          "Limit (SAR)": "SAR 1,500"
        },
        {
          "Benefit": "Dialysis",
          "Benefit Description": "Renal replacement therapy",
          "Covered Services": "In-center, peritoneal, or home dialysis",
          "Limit (SAR)": "SAR 180,000"
        },
        {
          "Benefit": "Kidney Transplant",
          "Benefit Description": "ESRD organ transplant",
          "Covered Services": "Recipient and donor procedure costs",
          "Limit (SAR)": "SAR 250,000"
        },
        {
          "Benefit": "Organ Harvest (Donor)",
          "Benefit Description": "Donor surgery",
          "Covered Services": "For insured donor",
          "Limit (SAR)": "SAR 50,000"
        },
        {
          "Benefit": "Home Healthcare",
          "Benefit Description": "At-home follow-up care",
          "Covered Services": "Post-surgery wound care, IV therapy, catheter support",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Telemedicine",
          "Benefit Description": "Remote consultations",
          "Covered Services": "MoH-licensed providers",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Medical Equipment",
          "Benefit Description": "Medical monitoring tools",
          "Covered Services": "Glucose monitors, insulin pumps, BP monitors, hearing aids",
          "Limit (SAR)": "Hearing aids SAR 6,000; others per policy"
        },
        {
          "Benefit": "Disability Treatment",
          "Benefit Description": "Rehabilitation and assistive support",
          "Covered Services": "Physical therapy, speech therapy, occupational therapy",
          "Limit (SAR)": "SAR 100,000"
        },
        {
          "Benefit": "Congenital/Hereditary Conditions",
          "Benefit Description": "Genetic or birth-related disorders",
          "Covered Services": "All congenital defects with life-threatening implications",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Circumcision (Male)",
          "Benefit Description": "Infant male circumcision",
          "Covered Services": "As per medical indication",
          "Limit (SAR)": "SAR 500"
        },
        {
          "Benefit": "Corpse Repatriation",
          "Benefit Description": "Repatriation of deceased to home country",
          "Covered Services": "Preparation and transport",
          "Limit (SAR)": "SAR 10,000"
        },
        {
          "Benefit": "Complications Coverage",
          "Benefit Description": "Post-treatment medical issues",
          "Covered Services": "From approved procedures",
          "Limit (SAR)": "Within overall policy cap"
        }
      ]
    },
    {
      "package_type": "B. Bronze Package",
      "general_info": {
        "Product": "B. Bronze Package",
        "Network": "S",
        "Annual Limit": "SAR 500,000",
        "Room": "Shared",
        "Deductible%": "20%",
        "Hospital deductible limit (Co-Pay)": "SAR 300",
        "Optical Limit": "SAR 400",
        "Dental deductible": "20%",
        "Dental Limit": "SAR 2,000",
        "Metarnity": "SAR 15,000"
      },
      "benefits": [
        {
          "Benefit": "Hospital Admission",
          "Benefit Description": "Inpatient care, surgeries, one-day operations, maternity",
          "Covered Services": "Shared room, surgeries, pregnancy, childbirth",
          "Limit (SAR)": "Covered, shared room"
        },
        {
          "Benefit": "Emergency Treatment",
          "Benefit Description": "Urgent care for life-threatening or disabling conditions",
          "Covered Services": "Resuscitation, emergency surgery, trauma",
          "Limit (SAR)": "Full coverage"
        },
        {
          "Benefit": "Outpatient Services",
          "Benefit Description": "General consultations, referrals, lab, x-rays",
          "Covered Services": "All specialties, family medicine, pediatrics",
          "Limit (SAR)": "Deductible 20%, Co-pay SAR 300"
        },
        {
          "Benefit": "Medication",
          "Benefit Description": "Coverage for prescription drugs",
          "Covered Services": "Generics: 20% copay (SAR 30 max); Brands (if no generic): 0–50%",
          "Limit (SAR)": "Per IDF policy"
        },
        {
          "Benefit": "Mental Health",
          "Benefit Description": "Treatment of psychological conditions",
          "Covered Services": "Depression, anxiety, schizophrenia, PTSD, autism, Alzheimer's",
          "Limit (SAR)": "Up to SAR 50,000 (where applicable)"
        },
        {
          "Benefit": "Maternity and Childcare",
          "Benefit Description": "Pre/postnatal care, childbirth, infant coverage",
          "Covered Services": "Delivery + complications, newborn up to 30 days",
          "Limit (SAR)": "SAR 15,000"
        },
        {
          "Benefit": "Newborn Screening",
          "Benefit Description": "Early detection of congenital disabilities",
          "Covered Services": "Critical congenital heart disease, metabolic screening",
          "Limit (SAR)": "SAR 100,000"
        },
        {
          "Benefit": "Vaccinations",
          "Benefit Description": "Preventive immunizations",
          "Covered Services": "Per MoH vaccination schedule for adults and children",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Dental Care",
          "Benefit Description": "Basic dental treatments",
          "Covered Services": "Consultations, cleaning, fillings, extractions, periodontal care",
          "Limit (SAR)": "SAR 2,000; 20% deductible"
        },
        {
          "Benefit": "Optical Care",
          "Benefit Description": "Vision correction",
          "Covered Services": "Eyeglasses for <14 yrs",
          "Limit (SAR)": "SAR 400"
        },
        {
          "Benefit": "Bariatric Surgery",
          "Benefit Description": "Obesity-related surgeries",
          "Covered Services": "If BMI ≥ 40 or ≥ 35 with comorbidities",
          "Limit (SAR)": "SAR 15,000 (as per national cap)"
        },
        {
          "Benefit": "Contraception",
          "Benefit Description": "Family planning services",
          "Covered Services": "Hormonal treatment, IUDs",
          "Limit (SAR)": "SAR 1,500"
        },
        {
          "Benefit": "Dialysis",
          "Benefit Description": "Renal replacement therapy",
          "Covered Services": "In-center, peritoneal, or home dialysis",
          "Limit (SAR)": "SAR 180,000"
        },
        {
          "Benefit": "Kidney Transplant",
          "Benefit Description": "ESRD organ transplant",
          "Covered Services": "Recipient and donor procedure costs",
          "Limit (SAR)": "SAR 250,000"
        },
        {
          "Benefit": "Organ Harvest (Donor)",
          "Benefit Description": "Donor surgery",
          "Covered Services": "For insured donor",
          "Limit (SAR)": "SAR 50,000"
        },
        {
          "Benefit": "Home Healthcare",
          "Benefit Description": "At-home follow-up care",
          "Covered Services": "Post-surgery wound care, IV therapy, catheter support",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Telemedicine",
          "Benefit Description": "Remote consultations",
          "Covered Services": "MoH-licensed providers",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Medical Equipment",
          "Benefit Description": "Medical monitoring tools",
          "Covered Services": "Glucose monitors, insulin pumps, BP monitors, hearing aids",
          "Limit (SAR)": "Hearing aids SAR 6,000; others per policy"
        },
        {
          "Benefit": "Disability Treatment",
          "Benefit Description": "Rehabilitation and assistive support",
          "Covered Services": "Physical therapy, speech therapy, occupational therapy",
          "Limit (SAR)": "SAR 100,000"
        },
        {
          "Benefit": "Congenital/Hereditary Conditions",
          "Benefit Description": "Genetic or birth-related disorders",
          "Covered Services": "All congenital defects with life-threatening implications",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Circumcision (Male)",
          "Benefit Description": "Infant male circumcision",
          "Covered Services": "As per medical indication",
          "Limit (SAR)": "SAR 500"
        },
        {
          "Benefit": "Corpse Repatriation",
          "Benefit Description": "Repatriation of deceased to home country",
          "Covered Services": "Preparation and transport",
          "Limit (SAR)": "SAR 10,000"
        },
        {
          "Benefit": "Complications Coverage",
          "Benefit Description": "Post-treatment medical issues",
          "Covered Services": "From approved procedures",
          "Limit (SAR)": "Within overall policy cap"
        }
      ]
    },
    {
      "package_type": "C. Silver Package",
      "general_info": {
        "Product": "C. Silver Package",
        "Network": "1",
        "Annual Limit": "SAR 500,000",
        "Room": "Shared",
        "Deductible%": "20%",
        "Hospital deductible limit (Co-Pay)": "SAR 300",
        "Optical Limit": "SAR 400",
        "Dental deductible": "20%",
        "Dental Limit": "SAR 2,000",
        "Metarnity": "SAR 15,000"
      },
      "benefits": [
        {
          "Benefit": "Hospital Admission",
          "Benefit Description": "Inpatient care, surgeries, one-day operations, maternity",
          "Covered Services": "Shared room, surgeries, pregnancy, childbirth",
          "Limit (SAR)": "Covered, shared room"
        },
        {
          "Benefit": "Emergency Treatment",
          "Benefit Description": "Urgent care for life-threatening or disabling conditions",
          "Covered Services": "Resuscitation, emergency surgery, trauma",
          "Limit (SAR)": "Full coverage"
        },
        {
          "Benefit": "Outpatient Services",
          "Benefit Description": "General consultations, referrals, lab, x-rays",
          "Covered Services": "All specialties, family medicine, pediatrics",
          "Limit (SAR)": "Deductible 20%, Co-pay:\nSAR 300 (Network 1) \nSAR 100 (Neworkt 3)"
        },
        {
          "Benefit": "Medication",
          "Benefit Description": "Coverage for prescription drugs",
          "Covered Services": "Generics: 20% copay (SAR 30 max); Brands (if no generic): 0–50%",
          "Limit (SAR)": "Per IDF policy"
        },
        {
          "Benefit": "Mental Health",
          "Benefit Description": "Treatment of psychological conditions",
          "Covered Services": "Depression, anxiety, schizophrenia, PTSD, autism, Alzheimer's",
          "Limit (SAR)": "Up to SAR 50,000"
        },
        {
          "Benefit": "Maternity and Childcare",
          "Benefit Description": "Pre/postnatal care, childbirth, infant coverage",
          "Covered Services": "Delivery + complications, newborn up to 30 days",
          "Limit (SAR)": "SAR 15,000"
        },
        {
          "Benefit": "Newborn Screening",
          "Benefit Description": "Early detection of congenital disabilities",
          "Covered Services": "Critical congenital heart disease, metabolic screening",
          "Limit (SAR)": "SAR 100,000"
        },
        {
          "Benefit": "Vaccinations",
          "Benefit Description": "Preventive immunizations",
          "Covered Services": "Per MoH vaccination schedule for adults and children",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Dental Care",
          "Benefit Description": "Basic dental treatments",
          "Covered Services": "Consultations, cleaning, fillings, extractions, periodontal care",
          "Limit (SAR)": "SAR 2,000; 20% deductible"
        },
        {
          "Benefit": "Optical Care",
          "Benefit Description": "Vision correction",
          "Covered Services": "Eyeglasses for <14 yrs",
          "Limit (SAR)": "SAR 400"
        },
        {
          "Benefit": "Bariatric Surgery",
          "Benefit Description": "Obesity-related surgeries",
          "Covered Services": "If BMI ≥ 40 or ≥ 35 with comorbidities",
          "Limit (SAR)": "SAR 15,000"
        },
        {
          "Benefit": "Contraception",
          "Benefit Description": "Family planning services",
          "Covered Services": "Hormonal treatment, IUDs",
          "Limit (SAR)": "SAR 1,500"
        },
        {
          "Benefit": "Dialysis",
          "Benefit Description": "Renal replacement therapy",
          "Covered Services": "In-center, peritoneal, or home dialysis",
          "Limit (SAR)": "SAR 180,000"
        },
        {
          "Benefit": "Kidney Transplant",
          "Benefit Description": "ESRD organ transplant",
          "Covered Services": "Recipient and donor procedure costs",
          "Limit (SAR)": "SAR 250,000"
        },
        {
          "Benefit": "Organ Harvest (Donor)",
          "Benefit Description": "Donor surgery",
          "Covered Services": "For insured donor",
          "Limit (SAR)": "SAR 50,000"
        },
        {
          "Benefit": "Home Healthcare",
          "Benefit Description": "At-home follow-up care",
          "Covered Services": "Post-surgery wound care, IV therapy, catheter support",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Telemedicine",
          "Benefit Description": "Remote consultations",
          "Covered Services": "MoH-licensed providers",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Medical Equipment",
          "Benefit Description": "Medical monitoring tools",
          "Covered Services": "Glucose monitors, insulin pumps, BP monitors, hearing aids",
          "Limit (SAR)": "Hearing aids SAR 6,000; others per policy"
        },
        {
          "Benefit": "Disability Treatment",
          "Benefit Description": "Rehabilitation and assistive support",
          "Covered Services": "Physical therapy, speech therapy, occupational therapy",
          "Limit (SAR)": "SAR 100,000"
        },
        {
          "Benefit": "Congenital/Hereditary Conditions",
          "Benefit Description": "Genetic or birth-related disorders",
          "Covered Services": "All congenital defects with life-threatening implications",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Circumcision (Male)",
          "Benefit Description": "Infant male circumcision",
          "Covered Services": "As per medical indication",
          "Limit (SAR)": "SAR 500"
        },
        {
          "Benefit": "Corpse Repatriation",
          "Benefit Description": "Repatriation of deceased to home country",
          "Covered Services": "Preparation and transport",
          "Limit (SAR)": "SAR 10,000"
        },
        {
          "Benefit": "Complications Coverage",
          "Benefit Description": "Post-treatment medical issues",
          "Covered Services": "From approved procedures",
          "Limit (SAR)": "Within overall policy cap"
        }
      ]
    },
    {
      "package_type": "C. Silver Package",
      "general_info": {
        "Product": "C. Silver Package",
        "Network": "3",
        "Annual Limit": "SAR 500,000",
        "Room": "Shared",
        "Deductible%": "20%",
        "Hospital deductible limit (Co-Pay)": "SAR 100",
        "Optical Limit": "SAR 400",
        "Dental deductible": "20%",
        "Dental Limit": "SAR 2,000",
        "Metarnity": "SAR 15,000"
      },
      "benefits": [
        {
          "Benefit": "Hospital Admission",
          "Benefit Description": "Inpatient care, surgeries, one-day operations, maternity",
          "Covered Services": "Shared room, surgeries, pregnancy, childbirth",
          "Limit (SAR)": "Covered, shared room"
        },
        {
          "Benefit": "Emergency Treatment",
          "Benefit Description": "Urgent care for life-threatening or disabling conditions",
          "Covered Services": "Resuscitation, emergency surgery, trauma",
          "Limit (SAR)": "Full coverage"
        },
        {
          "Benefit": "Outpatient Services",
          "Benefit Description": "General consultations, referrals, lab, x-rays",
          "Covered Services": "All specialties, family medicine, pediatrics",
          "Limit (SAR)": "Deductible 20%, Co-pay:\nSAR 300 (Network 1) \nSAR 100 (Neworkt 3)"
        },
        {
          "Benefit": "Medication",
          "Benefit Description": "Coverage for prescription drugs",
          "Covered Services": "Generics: 20% copay (SAR 30 max); Brands (if no generic): 0–50%",
          "Limit (SAR)": "Per IDF policy"
        },
        {
          "Benefit": "Mental Health",
          "Benefit Description": "Treatment of psychological conditions",
          "Covered Services": "Depression, anxiety, schizophrenia, PTSD, autism, Alzheimer's",
          "Limit (SAR)": "Up to SAR 50,000"
        },
        {
          "Benefit": "Maternity and Childcare",
          "Benefit Description": "Pre/postnatal care, childbirth, infant coverage",
          "Covered Services": "Delivery + complications, newborn up to 30 days",
          "Limit (SAR)": "SAR 15,000"
        },
        {
          "Benefit": "Newborn Screening",
          "Benefit Description": "Early detection of congenital disabilities",
          "Covered Services": "Critical congenital heart disease, metabolic screening",
          "Limit (SAR)": "SAR 100,000"
        },
        {
          "Benefit": "Vaccinations",
          "Benefit Description": "Preventive immunizations",
          "Covered Services": "Per MoH vaccination schedule for adults and children",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Dental Care",
          "Benefit Description": "Basic dental treatments",
          "Covered Services": "Consultations, cleaning, fillings, extractions, periodontal care",
          "Limit (SAR)": "SAR 2,000; 20% deductible"
        },
        {
          "Benefit": "Optical Care",
          "Benefit Description": "Vision correction",
          "Covered Services": "Eyeglasses for <14 yrs",
          "Limit (SAR)": "SAR 400"
        },
        {
          "Benefit": "Bariatric Surgery",
          "Benefit Description": "Obesity-related surgeries",
          "Covered Services": "If BMI ≥ 40 or ≥ 35 with comorbidities",
          "Limit (SAR)": "SAR 15,000"
        },
        {
          "Benefit": "Contraception",
          "Benefit Description": "Family planning services",
          "Covered Services": "Hormonal treatment, IUDs",
          "Limit (SAR)": "SAR 1,500"
        },
        {
          "Benefit": "Dialysis",
          "Benefit Description": "Renal replacement therapy",
          "Covered Services": "In-center, peritoneal, or home dialysis",
          "Limit (SAR)": "SAR 180,000"
        },
        {
          "Benefit": "Kidney Transplant",
          "Benefit Description": "ESRD organ transplant",
          "Covered Services": "Recipient and donor procedure costs",
          "Limit (SAR)": "SAR 250,000"
        },
        {
          "Benefit": "Organ Harvest (Donor)",
          "Benefit Description": "Donor surgery",
          "Covered Services": "For insured donor",
          "Limit (SAR)": "SAR 50,000"
        },
        {
          "Benefit": "Home Healthcare",
          "Benefit Description": "At-home follow-up care",
          "Covered Services": "Post-surgery wound care, IV therapy, catheter support",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Telemedicine",
          "Benefit Description": "Remote consultations",
          "Covered Services": "MoH-licensed providers",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Medical Equipment",
          "Benefit Description": "Medical monitoring tools",
          "Covered Services": "Glucose monitors, insulin pumps, BP monitors, hearing aids",
          "Limit (SAR)": "Hearing aids SAR 6,000; others per policy"
        },
        {
          "Benefit": "Disability Treatment",
          "Benefit Description": "Rehabilitation and assistive support",
          "Covered Services": "Physical therapy, speech therapy, occupational therapy",
          "Limit (SAR)": "SAR 100,000"
        },
        {
          "Benefit": "Congenital/Hereditary Conditions",
          "Benefit Description": "Genetic or birth-related disorders",
          "Covered Services": "All congenital defects with life-threatening implications",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Circumcision (Male)",
          "Benefit Description": "Infant male circumcision",
          "Covered Services": "As per medical indication",
          "Limit (SAR)": "SAR 500"
        },
        {
          "Benefit": "Corpse Repatriation",
          "Benefit Description": "Repatriation of deceased to home country",
          "Covered Services": "Preparation and transport",
          "Limit (SAR)": "SAR 10,000"
        },
        {
          "Benefit": "Complications Coverage",
          "Benefit Description": "Post-treatment medical issues",
          "Covered Services": "From approved procedures",
          "Limit (SAR)": "Within overall policy cap"
        }
      ]
    },
    {
      "package_type": "D. Gold Package",
      "general_info": {
        "Product": "D. Gold Package",
        "Network": "6",
        "Annual Limit": "SAR 500,000",
        "Room": "Private",
        "Deductible%": "20%",
        "Hospital deductible limit (Co-Pay)": "SAR 100",
        "Optical Limit": "SAR 1,500",
        "Dental deductible": "20%",
        "Dental Limit": "SAR 2,000",
        "Metarnity": "SAR 15,000"
      },
      "benefits": [
        {
          "Benefit": "Hospital Admission",
          "Benefit Description": "Inpatient care, surgeries, one-day operations, maternity",
          "Covered Services": "Private room, surgeries, pregnancy, childbirth",
          "Limit (SAR)": "Covered, private room"
        },
        {
          "Benefit": "Emergency Treatment",
          "Benefit Description": "Urgent care for life-threatening or disabling conditions",
          "Covered Services": "Resuscitation, emergency surgery, trauma",
          "Limit (SAR)": "Full coverage"
        },
        {
          "Benefit": "Outpatient Services",
          "Benefit Description": "General consultations, referrals, lab, x-rays",
          "Covered Services": "All specialties, family medicine, pediatrics",
          "Limit (SAR)": "Deductible 20%, Co-pay SAR 100"
        },
        {
          "Benefit": "Medication",
          "Benefit Description": "Coverage for prescription drugs",
          "Covered Services": "Generics: 20% copay (SAR 30 max); Brands (if no generic): 0–50%",
          "Limit (SAR)": "Per IDF policy"
        },
        {
          "Benefit": "Mental Health",
          "Benefit Description": "Treatment of psychological conditions",
          "Covered Services": "Depression, anxiety, schizophrenia, PTSD, autism, Alzheimer's",
          "Limit (SAR)": "Up to SAR 50,000"
        },
        {
          "Benefit": "Maternity and Childcare",
          "Benefit Description": "Pre/postnatal care, childbirth, infant coverage",
          "Covered Services": "Delivery + complications, newborn up to 30 days",
          "Limit (SAR)": "SAR 15,000"
        },
        {
          "Benefit": "Newborn Screening",
          "Benefit Description": "Early detection of congenital disabilities",
          "Covered Services": "Critical congenital heart disease, metabolic screening",
          "Limit (SAR)": "SAR 100,000"
        },
        {
          "Benefit": "Vaccinations",
          "Benefit Description": "Preventive immunizations",
          "Covered Services": "Per MoH vaccination schedule for adults and children",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Dental Care",
          "Benefit Description": "Basic dental treatments",
          "Covered Services": "Consultations, cleaning, fillings, extractions, periodontal care",
          "Limit (SAR)": "SAR 2,000; 20% deductible"
        },
        {
          "Benefit": "Optical Care",
          "Benefit Description": "Vision correction",
          "Covered Services": "Eyeglasses for <14 yrs",
          "Limit (SAR)": "SAR 1,500"
        },
        {
          "Benefit": "Bariatric Surgery",
          "Benefit Description": "Obesity-related surgeries",
          "Covered Services": "If BMI ≥ 40 or ≥ 35 with comorbidities",
          "Limit (SAR)": "SAR 15,000"
        },
        {
          "Benefit": "Contraception",
          "Benefit Description": "Family planning services",
          "Covered Services": "Hormonal treatment, IUDs",
          "Limit (SAR)": "SAR 1,500"
        },
        {
          "Benefit": "Dialysis",
          "Benefit Description": "Renal replacement therapy",
          "Covered Services": "In-center, peritoneal, or home dialysis",
          "Limit (SAR)": "SAR 180,000"
        },
        {
          "Benefit": "Kidney Transplant",
          "Benefit Description": "ESRD organ transplant",
          "Covered Services": "Recipient and donor procedure costs",
          "Limit (SAR)": "SAR 250,000"
        },
        {
          "Benefit": "Organ Harvest (Donor)",
          "Benefit Description": "Donor surgery",
          "Covered Services": "For insured donor",
          "Limit (SAR)": "SAR 50,000"
        },
        {
          "Benefit": "Home Healthcare",
          "Benefit Description": "At-home follow-up care",
          "Covered Services": "Post-surgery wound care, IV therapy, catheter support",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Telemedicine",
          "Benefit Description": "Remote consultations",
          "Covered Services": "MoH-licensed providers",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Medical Equipment",
          "Benefit Description": "Medical monitoring tools",
          "Covered Services": "Glucose monitors, insulin pumps, BP monitors, hearing aids",
          "Limit (SAR)": "Hearing aids SAR 6,000; others per policy"
        },
        {
          "Benefit": "Disability Treatment",
          "Benefit Description": "Rehabilitation and assistive support",
          "Covered Services": "Physical therapy, speech therapy, occupational therapy",
          "Limit (SAR)": "SAR 100,000"
        },
        {
          "Benefit": "Congenital/Hereditary Conditions",
          "Benefit Description": "Genetic or birth-related disorders",
          "Covered Services": "All congenital defects with life-threatening implications",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Circumcision (Male)",
          "Benefit Description": "Infant male circumcision",
          "Covered Services": "As per medical indication",
          "Limit (SAR)": "SAR 500"
        },
        {
          "Benefit": "Corpse Repatriation",
          "Benefit Description": "Repatriation of deceased to home country",
          "Covered Services": "Preparation and transport",
          "Limit (SAR)": "SAR 10,000"
        },
        {
          "Benefit": "Complications Coverage",
          "Benefit Description": "Post-treatment medical issues",
          "Covered Services": "From approved procedures",
          "Limit (SAR)": "Within overall policy cap"
        }
      ]
    },
    {
      "package_type": "E. Diamond Pacakge",
      "general_info": {
        "Product": "E. Diamond Pacakge",
        "Network": "7",
        "Annual Limit": "SAR 500,000",
        "Room": "Private",
        "Deductible%": "20%",
        "Hospital deductible limit (Co-Pay)": "SAR 100",
        "Optical Limit": "SAR 2,000",
        "Dental deductible": "Nill",
        "Dental Limit": "SAR 2,000",
        "Metarnity": "SAR 15,000"
      },
      "benefits": [
        {
          "Benefit": "Hospital Admission",
          "Benefit Description": "Inpatient care, surgeries, one-day operations, maternity",
          "Covered Services": "Private room, surgeries, pregnancy, childbirth",
          "Limit (SAR)": "Covered, private room"
        },
        {
          "Benefit": "Emergency Treatment",
          "Benefit Description": "Urgent care for life-threatening or disabling conditions",
          "Covered Services": "Resuscitation, emergency surgery, trauma",
          "Limit (SAR)": "Full coverage"
        },
        {
          "Benefit": "Outpatient Services",
          "Benefit Description": "General consultations, referrals, lab, x-rays",
          "Covered Services": "All specialties, family medicine, pediatrics",
          "Limit (SAR)": "Deductible 20%, Co-pay SAR 100"
        },
        {
          "Benefit": "Medication",
          "Benefit Description": "Coverage for prescription drugs",
          "Covered Services": "Generics: 20% copay (SAR 30 max); Brands (if no generic): 0–50%",
          "Limit (SAR)": "Per IDF policy"
        },
        {
          "Benefit": "Mental Health",
          "Benefit Description": "Treatment of psychological conditions",
          "Covered Services": "Depression, anxiety, schizophrenia, PTSD, autism, Alzheimer's",
          "Limit (SAR)": "Up to SAR 50,000"
        },
        {
          "Benefit": "Maternity and Childcare",
          "Benefit Description": "Pre/postnatal care, childbirth, infant coverage",
          "Covered Services": "Delivery + complications, newborn up to 30 days",
          "Limit (SAR)": "SAR 15,000"
        },
        {
          "Benefit": "Newborn Screening",
          "Benefit Description": "Early detection of congenital disabilities",
          "Covered Services": "Critical congenital heart disease, metabolic screening",
          "Limit (SAR)": "SAR 100,000"
        },
        {
          "Benefit": "Vaccinations",
          "Benefit Description": "Preventive immunizations",
          "Covered Services": "Per MoH vaccination schedule for adults and children",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Dental Care",
          "Benefit Description": "Basic dental treatments",
          "Covered Services": "Consultations, cleaning, fillings, extractions, periodontal care",
          "Limit (SAR)": "SAR 2,000; 0% deductible (Nill)"
        },
        {
          "Benefit": "Optical Care",
          "Benefit Description": "Vision correction",
          "Covered Services": "Eyeglasses for <14 yrs",
          "Limit (SAR)": "SAR 2,000"
        },
        {
          "Benefit": "Bariatric Surgery",
          "Benefit Description": "Obesity-related surgeries",
          "Covered Services": "If BMI ≥ 40 or ≥ 35 with comorbidities",
          "Limit (SAR)": "SAR 15,000"
        },
        {
          "Benefit": "Contraception",
          "Benefit Description": "Family planning services",
          "Covered Services": "Hormonal treatment, IUDs",
          "Limit (SAR)": "SAR 1,500"
        },
        {
          "Benefit": "Dialysis",
          "Benefit Description": "Renal replacement therapy",
          "Covered Services": "In-center, peritoneal, or home dialysis",
          "Limit (SAR)": "SAR 180,000"
        },
        {
          "Benefit": "Kidney Transplant",
          "Benefit Description": "ESRD organ transplant",
          "Covered Services": "Recipient and donor procedure costs",
          "Limit (SAR)": "SAR 250,000"
        },
        {
          "Benefit": "Organ Harvest (Donor)",
          "Benefit Description": "Donor surgery",
          "Covered Services": "For insured donor",
          "Limit (SAR)": "SAR 50,000"
        },
        {
          "Benefit": "Home Healthcare",
          "Benefit Description": "At-home follow-up care",
          "Covered Services": "Post-surgery wound care, IV therapy, catheter support",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Telemedicine",
          "Benefit Description": "Remote consultations",
          "Covered Services": "MoH-licensed providers",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Medical Equipment",
          "Benefit Description": "Medical monitoring tools",
          "Covered Services": "Glucose monitors, insulin pumps, BP monitors, hearing aids",
          "Limit (SAR)": "Hearing aids SAR 6,000; others per policy"
        },
        {
          "Benefit": "Disability Treatment",
          "Benefit Description": "Rehabilitation and assistive support",
          "Covered Services": "Physical therapy, speech therapy, occupational therapy",
          "Limit (SAR)": "SAR 100,000"
        },
        {
          "Benefit": "Congenital/Hereditary Conditions",
          "Benefit Description": "Genetic or birth-related disorders",
          "Covered Services": "All congenital defects with life-threatening implications",
          "Limit (SAR)": "Covered"
        },
        {
          "Benefit": "Circumcision (Male)",
          "Benefit Description": "Infant male circumcision",
          "Covered Services": "As per medical indication",
          "Limit (SAR)": "SAR 500"
        },
        {
          "Benefit": "Corpse Repatriation",
          "Benefit Description": "Repatriation of deceased to home country",
          "Covered Services": "Preparation and transport",
          "Limit (SAR)": "SAR 10,000"
        },
        {
          "Benefit": "Complications Coverage",
          "Benefit Description": "Post-treatment medical issues",
          "Covered Services": "From approved procedures",
          "Limit (SAR)": "Within overall policy cap"
        }
      ]
    }
  ]
}



# === 3) Matching helpers ===
def _normalize(s: str) -> str:
    return s.strip().lower()

def _is_match(pkg_label: str, target_choice: str) -> bool:
    """Match 'basic/bronze/silver/gold/diamond' to labels like 'A. Basic Package'."""
    pl = _normalize(pkg_label)
    t = _normalize(target_choice)

    if t == "basic":
        return "basic" in pl
    if t == "bronze":
        return "bronze" in pl
    if t == "silver":
        return "silver" in pl
    if t == "gold":
        return "gold" in pl
    if t == "diamond":
        # Handle the sheet typo "Pacakge" just in case
        return ("diamond" in pl) or ("pacakge" in pl and "diamond" in pl)
    return False


# === 4) Tool ===
@tool("get_policy_package_details", args_schema=Policy)
def get_policy_package_details(package_type: str):
    """Provide details about a specific policy package (returns general_info + benefits)."""
    if "insurance_packages" not in insurance_data:
        return {"error": "insurance_data missing 'insurance_packages'."}

    matches = [
        pkg for pkg in insurance_data["insurance_packages"]
        if _is_match(pkg.get("package_type", ""), package_type)
    ]

    if not matches:
        return {
            "message": f"No packages matched '{package_type}'.",
            "available": [p.get("package_type", "") for p in insurance_data["insurance_packages"]]
        }

    return matches


# === 5) Agent factory ===
def agent_policy_package_details(llm):
    return create_react_agent(
        model=llm,
        prompt="You provide complete info about a specific insurance policy package, including general info and all benefits.",
        name="policy_detail_assistant",
        tools=[get_policy_package_details],
    )

import json
from pathlib import Path

from app import get_sql_connection

OUTPUT_FILE = Path(__file__).parent / "database_metadata.json"
CURATED_METADATA_FILE = (
    Path(__file__).parent / "curated_metadata.json"
)


# Business descriptions inferred from table names and schema.
# Verify ambiguous definitions against authoritative documentation.
TABLE_DESCRIPTIONS = {
    "DM_COUNTRY": "Country dimension containing country codes, names, and geographic regions.",
    "DM_DATE": "Date dimension containing calendar attributes for date-based analysis.",
    "DM_EMPLOYEE": "Employee dimension containing employee identifiers, contact details, location, and position.",
    "DM_ORDER_METHOD": "Dimension describing the methods through which orders are placed.",
    "DM_ORG": "Organization dimension containing organization codes and names.",
    "DM_PRODUCT": "Product dimension containing product identifiers, hierarchy, names, colors, and brands.",
    "DM_RETAILER": "Retailer dimension containing retailer identifiers, names, types, and country information.",
    "DM_RETAILER_SITE": "Retailer-site dimension containing site contact, address, and location details.",
    "FCT_SALES": (
        "Fact table containing sales-related records linked to date, "
        "organization, employee, retailer, retailer site, product, "
        "and order-method dimensions. Row grain and calculation rules "
        "require verification."
    )
}


COLUMN_DESCRIPTIONS = {
    "DM_COUNTRY": {
        "COUNTRY_KEY": "Warehouse key identifying a country record.",
        "COUNTRY_CODE": "Numeric country code; coding standard requires verification.",
        "COUNTRY_CODE_3_LETTER": "Three-letter country code.",
        "COUNTRY_CODE_2_LETTER": "Two-letter country code.",
        "REGION": "Geographic region associated with the country.",
        "COUNTRY": "Country name.",
    },
    "DM_DATE": {
        "DAY_KEY": "Key identifying a calendar-day record.",
        "DAY_DATE": "Calendar date represented by the record.",
        "CURRENT_YEAR": "Year associated with the calendar date.",
        "QUARTER_KEY": "Quarter key or code; encoding requires verification.",
        "CURRENT_QUARTER": "Quarter label associated with the date.",
        "MONTH_KEY": "Month key or code; encoding requires verification.",
        "MONTH_NAME": "Full month name.",
        "MONTH_SHORT_NAME": "Abbreviated month name.",
        "MONTH_LETTER": "Single-character month label.",
        "DAYS_IN_MONTH": "Number of days in the month.",
        "WEEK_OF_MONTH": "Week-of-month value; encoding requires verification.",
        "DAY_OF_WEEK": "Numeric day-of-week value; convention requires verification.",
        "WEEKDAY": "Name of the day of the week.",
        "IS_WEEKEND": "Flag indicating whether the date is considered a weekend.",
    },
    "DM_EMPLOYEE": {
        "EMPLOYEE_KEY": "Warehouse key identifying an employee record.",
        "EMPLOYEE_CODE": "Employee business or source-system identifier.",
        "EMPLOYEE_NAME": "Employee name.",
        "ADDRESS1": "First line of the employee address.",
        "ADDRESS2": "Second line of the employee address.",
        "CITY": "Employee city.",
        "PROV_STATE": "Employee province or state.",
        "GENDER": "Gender value recorded for the employee.",
        "WORK_PHONE": "Employee work telephone number.",
        "EMAIL": "Employee email address.",
        "POSITION": "Employee job title or position.",
    },
    "DM_ORDER_METHOD": {
        "ORDER_METHOD_KEY": "Warehouse key identifying an order method.",
        "ORDER_METHOD_CODE": "Business or source-system code for the order method.",
        "ORDER_METHOD": "Order method name or description.",
    },
    "DM_ORG": {
        "ORGANIZATION_KEY": "Warehouse key identifying an organization.",
        "ORGANIZATION_CODE": "Business or source-system organization code.",
        "ORGANIZATION_NAME": "Organization name.",
    },
    "DM_PRODUCT": {
        "PRODUCT_KEY": "Warehouse key identifying a product.",
        "PRODUCT_LINE_CODE": "Code identifying the product line.",
        "PRODUCT_LINE": "High-level product category or line.",
        "PRODUCT_TYPE_CODE": "Code identifying the product type.",
        "PRODUCT_TYPE": "Product type or subcategory.",
        "PRODUCT_NUMBER": "Product business or source-system number.",
        "PRODUCT_NAME": "Product name.",
        "PRODUCT_COLOR": "Recorded product color.",
        "PRODUCT_BRAND": "Product brand.",
    },
    "DM_RETAILER": {
        "RETAILER_KEY": "Warehouse key identifying a retailer.",
        "RETAILER_CODE": "Business or source-system retailer code.",
        "RETAILER_NAME": "Retailer name.",
        "RETAILER_TYPE": "Retailer classification.",
        "RETAILER_COUNTRY_CODE": "Country code associated with the retailer; relationship requires verification.",
    },
    "DM_RETAILER_SITE": {
        "RETAILER_SITE_KEY": "Warehouse key identifying a retailer site.",
        "RETAILER_SITE_CODE": "Business or source-system site code.",
        "RETAILER_SITE_CONTACT_NAME": "Retailer-site contact person's name.",
        "RETAILER_SITE_PHONE_NUMBER": "Retailer-site telephone number.",
        "RETAILER_SITE_EMAIL": "Retailer-site email address.",
        "RETAILER_SITE_ADDRESS1": "First line of the retailer-site address.",
        "RETAILER_SITE_ADDRESS2": "Second line of the retailer-site address.",
        "RETAILER_SITE_CITY": "City where the retailer site is located.",
        "RETAILER_SITE_PROV_STATE": "Province or state where the site is located.",
        "RETAILER_SITE_COUNTRY_CODE": "Country code associated with the site; relationship requires verification.",
        "RETAILER_SITE_LAT": "Latitude stored as text; validate before numeric calculations.",
        "RETAILER_SITE_LON": "Longitude stored as text; validate before numeric calculations.",
    },
    "FCT_SALES": {
        "ORDER_DAY_KEY": "Order-date key referencing dbo.DM_DATE.DAY_KEY.",
        "SHIP_DAY_KEY": "Shipping-date key referencing dbo.DM_DATE.DAY_KEY.",
        "CLOSE_DAY_KEY": "Close-date key referencing dbo.DM_DATE.DAY_KEY.",
        "ORGANIZATION_KEY": "Organization key referencing dbo.DM_ORG.ORGANIZATION_KEY.",
        "EMPLOYEE_KEY": "Employee key referencing dbo.DM_EMPLOYEE.EMPLOYEE_KEY.",
        "RETAILER_KEY": "Retailer key referencing dbo.DM_RETAILER.RETAILER_KEY.",
        "RETAILER_SITE_KEY": "Retailer-site key referencing dbo.DM_RETAILER_SITE.RETAILER_SITE_KEY.",
        "PRODUCT_KEY": "Product key referencing dbo.DM_PRODUCT.PRODUCT_KEY.",
        "ORDER_METHOD_KEY": "Order-method key referencing dbo.DM_ORDER_METHOD.ORDER_METHOD_KEY.",
        "SALES_QUANTITY": "Sales quantity measure; units and business rules require verification.",
        "UNIT_CODE": "Ambiguous decimal field; business meaning requires verification.",
        "UNIT_SALE_PRICE": "Recorded unit sale price; currency, discounts, and revenue rules require verification.",
    }
}


def generate_metadata():
    metadata = {
        "database": "GOSALESDW_SIM",
        "tables": [],
        "logical_relationships" : [],
        "metadata_notes": [
            "Technical schema details and declared relationships are extracted from SQL Server.",
            "Descriptions inferred from names and schema context must be verified where business meaning is ambiguous.",
            "The exact grain of FCT_SALES and the official sales calculation are not established by schema metadata alone.",
            "Do not assume UNIT_CODE is a sales amount.",
            "Only declared foreign-key relationships are automatically extracted; additional logical relationships may need documentation.",
        ],
    }

    with get_sql_connection() as connection:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                s.name AS schema_name,
                t.name AS table_name,
                CAST(ep.value AS NVARCHAR(MAX)) AS table_description
            FROM sys.tables AS t
            INNER JOIN sys.schemas AS s
                ON s.schema_id = t.schema_id
            LEFT JOIN sys.extended_properties AS ep
                ON ep.major_id = t.object_id
                AND ep.minor_id = 0
                AND ep.name = 'MS_Description'
            ORDER BY s.name, t.name;
        """)

        table_rows = cursor.fetchall()

        for table_row in table_rows:
            schema_name = table_row.schema_name
            table_name = table_row.table_name
            qualified_name = f"{schema_name}.{table_name}"

            cursor.execute("""
                SELECT
                    c.name AS column_name,
                    ty.name AS data_type,
                    c.max_length,
                    c.precision,
                    c.scale,
                    c.is_nullable,
                    c.is_identity,
                    CAST(ep.value AS NVARCHAR(MAX)) AS column_description,
                    CASE WHEN pk.column_id IS NOT NULL
                         THEN 1 ELSE 0 END AS is_primary_key
                FROM sys.columns AS c
                INNER JOIN sys.types AS ty
                    ON c.user_type_id = ty.user_type_id
                LEFT JOIN sys.extended_properties AS ep
                    ON ep.major_id = c.object_id
                    AND ep.minor_id = c.column_id
                    AND ep.name = 'MS_Description'
                LEFT JOIN (
                    SELECT ic.object_id, ic.column_id
                    FROM sys.indexes AS i
                    INNER JOIN sys.index_columns AS ic
                        ON i.object_id = ic.object_id
                        AND i.index_id = ic.index_id
                    WHERE i.is_primary_key = 1
                ) AS pk
                    ON pk.object_id = c.object_id
                    AND pk.column_id = c.column_id
                WHERE c.object_id = OBJECT_ID(?)
                ORDER BY c.column_id;
            """, qualified_name)

            column_rows = cursor.fetchall()
            columns = []

            for col in column_rows:
                max_length = col.max_length

                if col.data_type in ("nvarchar", "nchar") and max_length > 0:
                    max_length //= 2

                inferred_description = COLUMN_DESCRIPTIONS.get(
                    table_name, {}
                ).get(
                    col.column_name,
                    f"Definition of {col.column_name} requires verification."
                )

                # Prefer existing SQL Server descriptions when available.
                description = (
                    col.column_description or inferred_description
                )

                columns.append({
                    "name": col.column_name,
                    "data_type": col.data_type,
                    "max_length": max_length if max_length >= 0 else None,
                    "precision": col.precision,
                    "scale": col.scale,
                    "nullable": bool(col.is_nullable),
                    "identity": bool(col.is_identity),
                    "primary_key": bool(col.is_primary_key),
                    "description": description,
                    "business_definition": description,
                    "definition_source": (
                        "sql_server_description"
                        if col.column_description
                        else "inferred_from_schema"
                    ),
                })

            cursor.execute("""
                SELECT
                    fk.name AS foreign_key_name,
                    pc.name AS source_column,
                    rs.name AS referenced_schema,
                    rt.name AS referenced_table,
                    rc.name AS referenced_column
                FROM sys.foreign_keys AS fk
                INNER JOIN sys.foreign_key_columns AS fkc
                    ON fk.object_id = fkc.constraint_object_id
                INNER JOIN sys.tables AS pt
                    ON fkc.parent_object_id = pt.object_id
                INNER JOIN sys.schemas AS ps
                    ON pt.schema_id = ps.schema_id
                INNER JOIN sys.columns AS pc
                    ON pc.object_id = pt.object_id
                    AND pc.column_id = fkc.parent_column_id
                INNER JOIN sys.tables AS rt
                    ON fkc.referenced_object_id = rt.object_id
                INNER JOIN sys.schemas AS rs
                    ON rt.schema_id = rs.schema_id
                INNER JOIN sys.columns AS rc
                    ON rc.object_id = rt.object_id
                    AND rc.column_id = fkc.referenced_column_id
                WHERE ps.name = ? AND pt.name = ?
                ORDER BY fk.name, fkc.constraint_column_id;
            """, schema_name, table_name)

            relationships = [
                {
                    "foreign_key_name": rel.foreign_key_name,
                    "column": rel.source_column,
                    "referenced_table": (
                        f"{rel.referenced_schema}.{rel.referenced_table}"
                    ),
                    "referenced_column": rel.referenced_column,
                }
                for rel in cursor.fetchall()
            ]

            table_description = (
                table_row.table_description
                or TABLE_DESCRIPTIONS.get(
                    table_name,
                    "Table definition requires verification."
                )
            )

            metadata["tables"].append({
                "schema": schema_name,
                "table_name": table_name,
                "description": table_description,
                "business_definition": table_description,
                "definition_source": (
                    "sql_server_description"
                    if table_row.table_description
                    else "inferred_from_schema"
                ),
                "columns": columns,
                "relationships": relationships,
            })

    # Load manually verified logical relationships.
        if CURATED_METADATA_FILE.exists():
            with CURATED_METADATA_FILE.open("r", encoding="utf-8") as file:
                curated_metadata = json.load(file)

            metadata["logical_relationships"] = curated_metadata.get(
                "logical_relationships", []
            )

        # Save the combined metadata.
        OUTPUT_FILE.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print("Metadata generation completed successfully.")
    print(f"Tables documented: {len(metadata['tables'])}")
    print(
        "Columns documented: "
        + str(sum(len(t["columns"]) for t in metadata["tables"]))
    )
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_metadata()

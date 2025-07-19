import openpyxl
import csv

def xlsx_to_csv(xlsx_filepath, csv_filepath, sheet_name_to_extract):
    # Load the workbook
    workbook = openpyxl.load_workbook(xlsx_filepath)
    
    # Iterate through each sheet
    for sheet_name in workbook.sheetnames:
        if sheet_name != sheet_name_to_extract:
            continue
        sheet = workbook[sheet_name]
        
        # Create CSV filename for the sheet
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['name', 'link'])
            for row in sheet.rows:
                for cell in row:
                    if cell is not None:
                        if cell.hyperlink is not None:
                            if cell.hyperlink.display is None:
                                link = cell.hyperlink.target
                            else:
                                link = cell.hyperlink.display
                            csv_writer.writerow([cell.value, link])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Extract hyperlinks from an Excel file and save them to a CSV file.')
    parser.add_argument('input_xlsx_file', type=str, help='Path to the input Excel file')
    parser.add_argument('output_csv_file', type=str, help='Path to the output CSV file')
    parser.add_argument('sheet_name', type=str, help='Name of the sheet to extract links from')
    args = parser.parse_args()
    xlsx_to_csv(args.input_xlsx_file, args.output_csv_file, args.sheet_name)

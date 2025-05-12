# === Compiler Settings ===
CXX = g++
CXXFLAGS = -std=c++20 -Wall -O3 -I.
EIGEN_FLAGS = -I/usr/include/eigen3

# === Paths ===
BINDIR := Bin
LOGDIR := logs
RESULTSDIR := results
STRUCTURE_DIR := hydrocarbon_structures

STRUCTURE_FILES := \
	$(STRUCTURE_DIR)/H2.txt \
	$(STRUCTURE_DIR)/CH4.txt \
	$(STRUCTURE_DIR)/C2H4.txt \
	$(STRUCTURE_DIR)/C2H6.txt \
	$(STRUCTURE_DIR)/C4H10.txt \
	$(STRUCTURE_DIR)/C6H6.txt

MOLECULES := H2 CH4 C2H4 C2H6 C4H10 C6H6

# === Build Targets ===
all: $(BINDIR)/cndos $(BINDIR)/mindo run_all extract_all combine_results

$(BINDIR)/cndos: cndos.cpp
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $(EIGEN_FLAGS) -o $@ $<

$(BINDIR)/mindo: mindo.cpp
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $(EIGEN_FLAGS) -o $@ $<

# === Run All Models ===
run_all: run_cndo2 run_cndos run_mindo

run_cndo2:
	@mkdir -p $(LOGDIR)
	@for file in $(STRUCTURE_FILES); do \
		name=$$(basename $$file .txt); \
		echo "Running CNDO/2 on $$file"; \
		$(BINDIR)/cndos $$file > $(LOGDIR)/$${name}_cndo2.log; \
	done

run_cndos:
	@mkdir -p $(LOGDIR)
	@for file in $(STRUCTURE_FILES); do \
		name=$$(basename $$file .txt); \
		echo "Running CNDO/S on $$file"; \
		$(BINDIR)/cndos $$file --overlap > $(LOGDIR)/$${name}_cndos.log; \
	done

run_mindo:
	@mkdir -p $(LOGDIR)
	@echo "Running MINDO..."
	@$(BINDIR)/mindo $(STRUCTURE_DIR)/H2.txt     1 1 MINDO    > $(LOGDIR)/H2_mindo.log
	@$(BINDIR)/mindo $(STRUCTURE_DIR)/CH4.txt    4 4 MINDO    > $(LOGDIR)/CH4_mindo.log
	@$(BINDIR)/mindo $(STRUCTURE_DIR)/C2H4.txt   6 6 MINDO    > $(LOGDIR)/C2H4_mindo.log
	@$(BINDIR)/mindo $(STRUCTURE_DIR)/C2H6.txt   7 7 MINDO    > $(LOGDIR)/C2H6_mindo.log
	@$(BINDIR)/mindo $(STRUCTURE_DIR)/C4H10.txt 13 13 MINDO   > $(LOGDIR)/C4H10_mindo.log
	@$(BINDIR)/mindo $(STRUCTURE_DIR)/C6H6.txt  15 15 MINDO   > $(LOGDIR)/C6H6_mindo.log
	@echo "MINDO calculations completed."
	@echo "Results saved in Results directory."

# === Extract CSVs from Logs ===
extract_all: extract_cndo2 extract_cndos extract_mindo

extract_cndo2:
	@mkdir -p $(RESULTSDIR)
	@echo "Molecule,Total Energy (eV),Nuclear Energy (eV),Electron Energy (eV)" > $(RESULTSDIR)/cndo2_results.csv
	@for file in $(LOGDIR)/*_cndo2.log; do \
		name=$$(basename $$file _cndo2.log); \
		total=$$(grep "Total Energy" $$file | grep -oP '[-+]?[0-9]*\.?[0-9]+'); \
		nuc=$$(grep "Nuclear" $$file | grep -oP '[-+]?[0-9]*\.?[0-9]+' | head -1); \
		elec=$$(grep "Electron" $$file | grep -oP '[-+]?[0-9]*\.?[0-9]+'); \
		echo "$$name,$$total,$$nuc,$$elec" >> $(RESULTSDIR)/cndo2_results.csv; \
	done

extract_cndos:
	@mkdir -p $(RESULTSDIR)
	@echo "Molecule,Total Energy (eV),Nuclear Energy (eV),Electron Energy (eV)" > $(RESULTSDIR)/cndos_results.csv
	@for file in $(LOGDIR)/*_cndos.log; do \
		name=$$(basename $$file _cndos.log); \
		total=$$(grep "Total Energy" $$file | grep -oP '[-+]?[0-9]*\.?[0-9]+'); \
		nuc=$$(grep "Nuclear" $$file | grep -oP '[-+]?[0-9]*\.?[0-9]+' | head -1); \
		elec=$$(grep "Electron" $$file | grep -oP '[-+]?[0-9]*\.?[0-9]+'); \
		echo "$$name,$$total,$$nuc,$$elec" >> $(RESULTSDIR)/cndos_results.csv; \
	done

extract_mindo:
	@mkdir -p $(RESULTSDIR)
	@echo "Molecule,Total Energy (eV),Nuclear Energy (eV),Electron Energy (eV)" > $(RESULTSDIR)/mindo_results.csv
	@for file in $(LOGDIR)/*_mindo.log; do \
		name=$$(basename $$file _mindo.log); \
		nuc=$$(grep "Nuclear Repulsion Energy" $$file | grep -oP '[-+]?[0-9]*\.?[0-9]+'); \
		elec=$$(grep "Electron Energy" $$file | grep -oP '[-+]?[0-9]*\.?[0-9]+'); \
		total=$$(awk "BEGIN {print $$nuc + $$elec}"); \
		echo "$$name,$$total,$$nuc,$$elec" >> $(RESULTSDIR)/mindo_results.csv; \
	done

# === Combine All Results ===
combine_results:
	@mkdir -p $(RESULTSDIR)
	@echo "Molecule,CNDO2_Total,CNDOS_Total,MINDO_Total" > $(RESULTSDIR)/combined_results.csv
	@for mol in $(MOLECULES); do \
		cndo2=$$(grep -m1 "$$mol" $(RESULTSDIR)/cndo2_results.csv | cut -d',' -f2); \
		cndos=$$(grep -m1 "$$mol" $(RESULTSDIR)/cndos_results.csv | cut -d',' -f2); \
		mindo=$$(grep -m1 "$$mol" $(RESULTSDIR)/mindo_results.csv | cut -d',' -f2); \
		echo "$$mol,$$cndo2,$$cndos,$$mindo" >> $(RESULTSDIR)/combined_results.csv; \
	done

# === Cleanup ===
clean:
	rm -rf $(BINDIR) $(LOGDIR) $(RESULTSDIR)

.PHONY: all run_all extract_all clean combine_results
